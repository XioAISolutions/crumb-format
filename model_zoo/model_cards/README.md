# Model Cards

This directory contains model cards for all Wave Field LLM models in the Model Zoo.

## Overview

Model cards provide comprehensive documentation for each pretrained model, including:

- Model architecture and specifications
- Training details and hyperparameters
- Performance benchmarks
- Usage examples
- Limitations and ethical considerations
- Citation information

## Available Model Cards

- [wavefield-tiny.md](wavefield-tiny.md) - Tiny model (230K params)
- [wavefield-small.md](wavefield-small.md) - Small model (25M params)
- [wavefield-medium.md](wavefield-medium.md) - Medium model (350M params)
- [wavefield-large.md](wavefield-large.md) - Large model (1.5B params)
- [wavefield-small-code.md](wavefield-small-code.md) - Code-specialized Small
- [wavefield-medium-crumb.md](wavefield-medium-crumb.md) - CRUMB-specialized Medium
- [wavefield-large-instruct.md](wavefield-large-instruct.md) - Instruction-tuned Large

## Generating Model Cards

Model cards are automatically generated from training artifacts using the `generator.py` script.

### Basic Usage

```bash
python -m model_zoo.model_cards.generator \
  --config model_zoo/configs/small.yaml \
  --training-log logs/wavefield-small/training.json \
  --eval-results results/wavefield-small-eval.json \
  --output model_zoo/model_cards/wavefield-small.md
```

### With Custom Metadata

```bash
python -m model_zoo.model_cards.generator \
  --config model_zoo/configs/small.yaml \
  --training-log logs/wavefield-small/training.json \
  --eval-results results/wavefield-small-eval.json \
  --output model_zoo/model_cards/wavefield-small.md \
  --license "Apache 2.0" \
  --authors "Wave Field LLM Team" \
  --hardware-used "8x NVIDIA A100 80GB" \
  --training-duration "7 days"
```

### Custom Template

You can use a custom template by specifying the `--template` argument:

```bash
python -m model_zoo.model_cards.generator \
  --config model_zoo/configs/small.yaml \
  --training-log logs/wavefield-small/training.json \
  --eval-results results/wavefield-small-eval.json \
  --output model_zoo/model_cards/wavefield-small.md \
  --template my_custom_template.md
```

## Template Format

The model card template uses placeholder variables in the format `{variable_name}`. Available variables include:

### Model Information
- `{model_name}` - Model name
- `{version}` - Model version
- `{description}` - Model description
- `{num_parameters}` - Number of parameters
- `{release_date}` - Release date

### Architecture
- `{dim}` - Hidden dimension
- `{n_layers}` - Number of layers
- `{n_heads}` - Number of attention heads
- `{field_size}` - Wave field size
- `{vocab_size}` - Vocabulary size
- `{max_seq_len}` - Maximum sequence length

### Training
- `{total_tokens}` - Total training tokens
- `{max_steps}` - Maximum training steps
- `{learning_rate}` - Learning rate
- `{training_duration}` - Training duration

### Performance
- `{benchmark_results_table}` - Benchmark results table
- `{val_perplexity}` - Validation perplexity
- `{inference_speed}` - Inference speed

See [template.md](template.md) for the complete list of variables.

## Model Card Structure

Each model card follows this structure:

1. **Model Description** - Overview and key details
2. **Model Details** - Architecture and training specifications
3. **Performance** - Benchmark results and metrics
4. **Usage** - Code examples and API usage
5. **Intended Use** - Primary use cases and limitations
6. **Limitations** - Known limitations and constraints
7. **Bias and Fairness** - Bias analysis and considerations
8. **Ethical Considerations** - Ethical guidelines
9. **Training Details** - Detailed training procedure
10. **Evaluation** - Evaluation methodology and results
11. **Environmental Impact** - Carbon footprint and sustainability
12. **Citation** - How to cite the model
13. **Additional Resources** - Links and references

## Best Practices

### When Creating Model Cards

1. **Be Comprehensive** - Include all relevant information
2. **Be Honest** - Document limitations and biases
3. **Be Specific** - Provide concrete examples and metrics
4. **Be Accessible** - Write for diverse audiences
5. **Be Current** - Update cards when models are updated

### Required Information

Every model card must include:

- Model architecture and size
- Training data and procedure
- Performance benchmarks
- Usage examples
- Known limitations
- Ethical considerations
- Citation information

### Optional Information

Consider including:

- Comparison with baseline models
- Ablation study results
- Failure case analysis
- Community feedback
- Version history

## Updating Model Cards

Model cards should be updated when:

- Model is retrained or fine-tuned
- New evaluation results are available
- Limitations or biases are discovered
- Usage patterns change
- Community provides feedback

To update a model card:

1. Regenerate using the generator script with updated artifacts
2. Manually edit for additional context
3. Increment the version number
4. Add entry to changelog section

## Contributing

To contribute a model card:

1. Train and evaluate your model
2. Generate the model card using the generator
3. Review and enhance with additional context
4. Submit a pull request with the model card

## Resources

- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)
- [Hugging Face Model Card Guide](https://huggingface.co/docs/hub/model-cards)
- [Google Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)

## License

Model cards are licensed under CC-BY-4.0, allowing reuse with attribution.