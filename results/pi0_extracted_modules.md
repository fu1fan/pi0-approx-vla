# pi0 Extracted Modules

This report records selected exact key matches. Missing entries are not fabricated.

## visual_projector
- selected: `paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight` shape `[2048, 1152]`
- selected: `paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias` shape `[2048]`
- extracted tensors: 2
## vlm_attention_layer0
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight` shape `[2048, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight` shape `[256, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.v_proj.weight` shape `[256, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.o_proj.weight` shape `[2048, 2048]`
- extracted tensors: 4
## vlm_attention_layer9
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.q_proj.weight` shape `[2048, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.k_proj.weight` shape `[256, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.v_proj.weight` shape `[256, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.self_attn.o_proj.weight` shape `[2048, 2048]`
- extracted tensors: 4
## vlm_ffn_layer0
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.weight` shape `[16384, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight` shape `[16384, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight` shape `[2048, 16384]`
- extracted tensors: 3
## vlm_ffn_layer9
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.gate_proj.weight` shape `[16384, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.up_proj.weight` shape `[16384, 2048]`
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.9.mlp.down_proj.weight` shape `[2048, 16384]`
- extracted tensors: 3
## action_expert_ffn_layer0
- selected: `paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj.weight` shape `[4096, 1024]`
- selected: `paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight` shape `[4096, 1024]`
- selected: `paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight` shape `[1024, 4096]`
- extracted tensors: 3
## action_projection
- selected: `state_proj.weight` shape `[1024, 32]`
- selected: `state_proj.bias` shape `[1024]`
- selected: `action_in_proj.weight` shape `[1024, 32]`
- selected: `action_in_proj.bias` shape `[1024]`
- selected: `action_out_proj.weight` shape `[32, 1024]`
- selected: `action_out_proj.bias` shape `[32]`
- extracted tensors: 6
## rmsnorm
- selected: `paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm.weight` shape `[2048]`
- selected: `paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight` shape `[1024]`
- extracted tensors: 2

selected tensor file: `results/pi0_module_weights/selected_modules.pt`
manifest: `results/pi0_module_weights/manifest.json`