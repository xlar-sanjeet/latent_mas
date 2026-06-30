import os
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    LLM = None
    SamplingParams = None
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Any]) -> int:
    if past_key_values is None:
        return 0

    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())

    try:
        if len(past_key_values) == 0:
            return 0

        return int(past_key_values[0][0].shape[-2])

    except (TypeError, IndexError, AttributeError) as exc:
        raise TypeError(
            f"Unsupported past_key_values type: "
            f"{type(past_key_values).__name__}"
        ) from exc


def _mean_embedding_norm(
    weight: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    total_norm = torch.zeros(
        (),
        device=weight.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for chunk in weight.detach().split(
            chunk_size,
            dim=0,
        ):
            total_norm += torch.linalg.vector_norm(
                chunk.float(),
                dim=1,
            ).sum()

    return total_norm / weight.shape[0]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.debug_probe = bool(getattr(args, "debug_probe", False)) if args else False
        self.args = args

        # Detect the model family once so callers can apply family-specific behavior.
        try:
            self.model_type = AutoConfig.from_pretrained(model_name).model_type
        except Exception:
            self.model_type = None

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            _ensure_pad_token(self.tokenizer)
            # Decoder-only batched inference requires left padding so the final
            # sequence position corresponds to the last real prompt token.
            self.tokenizer.padding_side = "left"
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self.HF_model.config.pad_token_id = self.tokenizer.pad_token_id
                self.HF_model.generation_config.pad_token_id = (
                    self.tokenizer.pad_token_id
                )
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        # Decoder-only batched inference requires left padding so the final
        # sequence position corresponds to the last real prompt token.
        self.tokenizer.padding_side = "left"
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if input_embeds is None:
            input_embeds = getattr(model, "embed_tokens", None)
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

        input_weight = input_embeds.weight
        output_weight = output_embeds.weight
        hidden_size = input_weight.shape[-1]
        target_device = torch.device(device)

        # The mean input-embedding norm is always required for norm restoration and
        # is computed in chunks to avoid a full FP32 copy of the embedding table.
        chunk_size = int(getattr(args, "realign_chunk_size", 4096))
        target_norm = _mean_embedding_norm(input_weight, chunk_size).to(target_device)

        realign_enabled = bool(getattr(args, "latent_space_realign", False))
        if not realign_enabled:
            # No Gram matrix, no linear solve, no full FP32 vocabulary copies:
            # simply return identity (still returning the mean-norm target above).
            realign_matrix = torch.eye(hidden_size, device=target_device, dtype=torch.float32)
            return realign_matrix, target_norm

        # Tied-embedding detection is used for diagnostics/logging only and must NOT
        # be used to skip matrix generation.
        weights_are_tied = (
            input_weight is output_weight
            or (
                input_weight.shape == output_weight.shape
                and input_weight.data_ptr() == output_weight.data_ptr()
            )
        )
        print(
            "[LatentMAS] Generating realignment matrix. "
            f"Tied embeddings: {weights_are_tied}"
        )

        # Accumulate the Gram matrix (Wout^T Wout) and RHS (Wout^T Ein) in
        # vocabulary chunks to avoid materializing full FP32 copies of the tables.
        gram = torch.zeros(
            hidden_size,
            hidden_size,
            device=target_device,
            dtype=torch.float32,
        )
        rhs = torch.zeros(
            hidden_size,
            hidden_size,
            device=target_device,
            dtype=torch.float32,
        )

        for start in range(0, input_weight.shape[0], chunk_size):
            end = min(start + chunk_size, input_weight.shape[0])

            input_chunk = input_weight[start:end].detach().to(
                device=target_device, dtype=torch.float32
            )
            output_chunk = output_weight[start:end].detach().to(
                device=target_device, dtype=torch.float32
            )

            gram.add_(output_chunk.T @ output_chunk)
            rhs.add_(output_chunk.T @ input_chunk)

            del input_chunk
            del output_chunk

        regularization_strength = float(getattr(args, "realign_regularization", 1e-5))
        regularization = regularization_strength * torch.eye(
            hidden_size,
            device=target_device,
            dtype=torch.float32,
        )

        # Solve (Wout^T Wout + lambda I) Wa = Wout^T Ein.
        realign_matrix = torch.linalg.solve(gram + regularization, rhs)

        if realign_matrix.shape != (hidden_size, hidden_size):
            raise RuntimeError("Unexpected realignment matrix shape.")

        if not torch.isfinite(realign_matrix).all():
            raise RuntimeError(
                "The generated realignment matrix contains "
                "NaN or infinite values."
            )

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        embedding_weight = model.get_input_embeddings().weight
        expected_hidden_size = embedding_weight.shape[-1]
        if hidden.shape[-1] != expected_hidden_size:
            raise ValueError(
                f"Hidden dimension {hidden.shape[-1]} does not "
                f"match embedding dimension "
                f"{expected_hidden_size}."
            )

        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = hidden_fp32 @ matrix

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)

        aligned = aligned.to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
        return aligned

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Any] = None,
        past_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], Optional[Any]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                if past_attention_mask is not None:
                    # Preserve zeros from previous padding positions.
                    past_mask = past_attention_mask.to(
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                else:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
        )
        sequences = outputs.sequences
        # With left-padded batches the generated tokens always begin at the full
        # padded input width, so slice from there rather than per-row real lengths.
        input_width = input_ids.shape[-1]
        generations: List[str] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, input_width:]
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Any] = None,
        past_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Any], torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        current_attention_mask = attention_mask
        if past_key_values is not None and _past_length(past_key_values) > 0:
            if past_attention_mask is not None:
                # Preserve zeros that represent previous padding positions.
                past_mask = past_attention_mask.to(
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            else:
                past_mask = torch.ones(
                    (attention_mask.shape[0], _past_length(past_key_values)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            full_attention_mask = torch.cat([past_mask, current_attention_mask], dim=-1)
        else:
            full_attention_mask = current_attention_mask

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        if self.debug_probe:
            self._probe_hidden_to_tokens(last_hidden, label="prompt-final-hidden")

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        batch_size = input_ids.shape[0]

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if self.debug_probe:
                self._probe_hidden_to_tokens(latent_vec, label=f"latent-step-{step}-input-embed")

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)
            if latent_embed.shape != (batch_size, 1, self.model.config.hidden_size):
                raise ValueError(
                    "Unexpected latent embedding shape "
                    f"{tuple(latent_embed.shape)}; expected "
                    f"{(batch_size, 1, self.model.config.hidden_size)}."
                )

            # Append exactly one valid mask position for this latent step.
            full_attention_mask = torch.cat(
                [
                    full_attention_mask,
                    torch.ones(
                        full_attention_mask.shape[0],
                        1,
                        device=full_attention_mask.device,
                        dtype=full_attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=full_attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            if self.debug_probe:
                self._probe_hidden_to_tokens(last_hidden, label=f"latent-step-{step}-output-hidden")

        return past, full_attention_mask

    @torch.no_grad()
    def _probe_hidden_to_tokens(self, hidden: torch.Tensor, label: str = "", topk: int = 5) -> None:
        # Decode a hidden/latent vector [B, D] to its nearest vocabulary tokens via
        # the output embedding (unembedding) so latent "thoughts" can be inspected.
        try:
            model = self.model
            output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
            if output_embeds is None:
                output_embeds = getattr(model, "lm_head", None)
            if output_embeds is None or not hasattr(output_embeds, "weight"):
                return
            weight = output_embeds.weight  # [V, D]
            h = hidden.detach().to(weight.dtype)
            logits = torch.matmul(h, weight.t())  # [B, V]
            probs = torch.softmax(logits.float(), dim=-1)
            top_probs, top_ids = probs.topk(topk, dim=-1)
            for b in range(h.shape[0]):
                toks = self.tokenizer.convert_ids_to_tokens(top_ids[b].tolist())
                pairs = ", ".join(
                    f"{tok!r}:{p:.3f}" for tok, p in zip(toks, top_probs[b].tolist())
                )
                print(f"[probe] {label} | row {b}: {pairs}")
        except Exception as exc:  # diagnostics must never crash the run
            print(f"[probe] {label}: failed ({exc})")
    @torch.no_grad()
    def append_token_to_past_batch(
        self,
        token_id: int,
        past_key_values: Any,
        past_attention_mask: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor]:
        # Feed a single token (e.g. the assistant end-of-turn marker) through the
        # model so the existing KV cache gains a proper turn boundary before the
        # next agent prompt is appended.
        batch_size = past_attention_mask.shape[0]
        token_ids = torch.full(
            (batch_size, 1), token_id, dtype=torch.long, device=self.device
        )
        new_mask = torch.ones(
            (batch_size, 1),
            dtype=past_attention_mask.dtype,
            device=past_attention_mask.device,
        )
        full_attention_mask = torch.cat([past_attention_mask, new_mask], dim=-1)
        outputs = self.model(
            input_ids=token_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.past_key_values, full_attention_mask

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

