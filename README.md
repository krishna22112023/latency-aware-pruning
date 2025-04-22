<p align="center">
<img src="assets/icon2.png" width="15%"> <br>
</p>

# *DiLAP*: Differentiable Latency-Aware Pruning

DiLAP is a framework for hardware-aware structured pruning of deep neural networks (agnostic to model type like ResNet/Transformers). It optimizes layer-wise pruning ratios to meet hardware latency constraints while maximizing model accuracy.

## Key features:

1. Differentiable Latency: Incorporates a differentiable latency predictor (trained for the target hardware) into the loss function, allowing gradient-based optimization against latency budgets.   
2. Importance Guidance: Uses Wanda-like scores (Weight magnitude * Activation norm) to calculate layer importance, guiding the pruning process.   
3. Bilevel Optimization: Employs a DARTS-inspired bilevel optimization approach to optimize pruning ratios (outer loop) based on validation loss and latency, while optimizing model weights (inner loop) based on training loss.   
4. Automated Ratio Finding: Automatically finds layer-specific pruning ratios instead of requiring manual specification.

## TO DO

Stage 1 : Pruning to compare with SOTA pruning methods
- [ ] Construct latency dataset for H100
- [ ] SOTA Methods include : 
    1. Magnitude only
    2. LLM-Streamline (2024)
    3. ShortGPT (2024)
    4. LoRAPrune (2024)
    5. WANDA (2023)
    6. LLM-Pruner (2023)
    7. Compresso (2023)
- [ ] Models used : llama2-7B
- [ ] Eval benchmark : WikiText2,PTB,MMLU (5-shot),OBQA,ARC-e,WinoGrande, ARC-c,PIQA,HellaSwag
- [ ] Eval metrics : Top-1 Accuracy

Stage 2 : Pruning to compare with latency constrained methods
- [ ] SOTA method : HALP (2022)
- [ ] Model used : Llama3.2-1B
- [ ] Eval benchmark : WikiText2,PTB,MMLU (5-shot),OBQA,ARC-e,WinoGrande, ARC-c,PIQA,HellaSwag
- [ ] Eval metrics : Top-1 Accuracy, FLOPs, Avg. Memory Util, Latency
- [ ] Devices : Jetson Orin GPU (Edge-GPU), Jetson Orin CPU (Edge-CPU) , Samsung S23 (phone), Snapdragon X Plus 8-Core CRD (laptop)
- [ ] Experiments : Different prune ratios (30%, 50%,80%)