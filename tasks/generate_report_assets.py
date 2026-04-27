import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('BMEAI_CONFIG')
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'config.yaml',
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def plot_dataset_split(split_counts, save_path):
    labels = list(split_counts)
    values = [split_counts[label] for label in labels]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=['#4E79A7', '#F28E2B', '#59A14F'])
    plt.title('Dataset Split by Slice Count')
    plt.ylabel('Number of 2D slices')
    plt.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f'{value}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(task2_summary, task3_summary, save_path):
    labels = ['Aliased Input', 'Task 2', 'Task 3']
    psnr_values = [
        task2_summary['psnr_before']['mean'],
        task2_summary['psnr_after']['mean'],
        task3_summary['psnr_after']['mean'],
    ]
    ssim_values = [
        task2_summary['ssim_before']['mean'],
        task2_summary['ssim_after']['mean'],
        task3_summary['ssim_after']['mean'],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, psnr_values, color=['#B0BEC5', '#4E79A7', '#E15759'])
    axes[0].set_title('PSNR Comparison')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(labels, ssim_values, color=['#B0BEC5', '#4E79A7', '#E15759'])
    axes[1].set_title('SSIM Comparison')
    axes[1].set_ylabel('SSIM')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_tables(config, split_counts, task2_summary=None, task3_summary=None):
    lines = []
    lines.append('# Generated Tables')
    lines.append('')
    lines.append('## Dataset Split')
    lines.append('')
    lines.append('| Split | Slice Count |')
    lines.append('| --- | ---: |')
    for split_name, count in split_counts.items():
        lines.append(f'| {split_name} | {count} |')
    lines.append('')

    lines.append('## Hyperparameters')
    lines.append('')
    lines.append('| Setting | Task 2 | Task 3 |')
    lines.append('| --- | --- | --- |')
    lines.append(f"| Input | Aliased T2 | Aliased T2 + full T1 |")
    lines.append(f"| Backbone | U-Net | Unrolled U-Net + DC |")
    lines.append(f"| Base channels | {config['task2']['base_channels']} | {config['task3']['base_channels']} |")
    lines.append(f"| Depth | {config['task2']['depth']} | {config['task3']['depth']} |")
    lines.append(f"| Batch size | {config['task2']['batch_size']} | {config['task3']['batch_size']} |")
    lines.append(f"| Epochs | {config['task2']['num_epochs']} | {config['task3']['num_epochs']} |")
    lines.append(f"| Learning rate | {config['task2']['learning_rate']} | {config['task3']['learning_rate']} |")
    lines.append(f"| Loss | MSE | {config['task3']['loss_type']} |")
    lines.append('')

    if task2_summary and task3_summary:
        lines.append('## Quantitative Results')
        lines.append('')
        lines.append('| Method | PSNR (dB) | SSIM |')
        lines.append('| --- | ---: | ---: |')
        lines.append(
            f"| Aliased input | {task2_summary['psnr_before']['mean']:.2f} | {task2_summary['ssim_before']['mean']:.4f} |"
        )
        lines.append(
            f"| Task 2 baseline | {task2_summary['psnr_after']['mean']:.2f} | {task2_summary['ssim_after']['mean']:.4f} |"
        )
        lines.append(
            f"| Task 3 multi-modal | {task3_summary['psnr_after']['mean']:.2f} | {task3_summary['ssim_after']['mean']:.4f} |"
        )
        lines.append('')
        lines.append('## Improvements Over Aliased Input')
        lines.append('')
        lines.append('| Method | PSNR Gain (dB) | SSIM Gain |')
        lines.append('| --- | ---: | ---: |')
        lines.append(
            f"| Task 2 baseline | {task2_summary['psnr_after']['mean'] - task2_summary['psnr_before']['mean']:.2f} | {task2_summary['ssim_after']['mean'] - task2_summary['ssim_before']['mean']:.4f} |"
        )
        lines.append(
            f"| Task 3 multi-modal | {task3_summary['psnr_after']['mean'] - task3_summary['psnr_before']['mean']:.2f} | {task3_summary['ssim_after']['mean'] - task3_summary['ssim_before']['mean']:.4f} |"
        )
        lines.append('')
        lines.append('## Task 3 Over Task 2')
        lines.append('')
        lines.append('| Comparison | Value |')
        lines.append('| --- | ---: |')
        lines.append(
            f"| PSNR gain | {task3_summary['psnr_after']['mean'] - task2_summary['psnr_after']['mean']:.2f} dB |"
        )
        lines.append(
            f"| SSIM gain | {task3_summary['ssim_after']['mean'] - task2_summary['ssim_after']['mean']:.4f} |"
        )
        lines.append('')

    return '\n'.join(lines)


def build_worst_case_table(worst_cases):
    lines = []
    lines.append('# Task 3 Worst Cases')
    lines.append('')
    lines.append('| Rank | Patient | Slice | PSNR Before | PSNR After | SSIM Before | SSIM After |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for index, case in enumerate(worst_cases, start=1):
        lines.append(
            f"| {index} | {case['patient']} | {case['slice_idx']} | "
            f"{case['psnr_before']:.2f} | {case['psnr_after']:.2f} | "
            f"{case['ssim_before']:.4f} | {case['ssim_after']:.4f} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main():
    config = load_config()
    output_dir = config['output']['dir']
    report_dir = os.path.join(output_dir, 'report_assets')
    os.makedirs(report_dir, exist_ok=True)

    train_ds, val_ds, test_ds, _ = create_dataloaders(config)
    split_counts = {
        'Train': len(train_ds),
        'Validation': len(val_ds),
        'Test': len(test_ds),
    }
    write_json(os.path.join(report_dir, 'dataset_split.json'), split_counts)
    plot_dataset_split(split_counts, os.path.join(report_dir, 'dataset_split.png'))

    task2_summary = None
    task3_summary = None
    task2_summary_path = os.path.join(output_dir, 'task2', 'evaluation_summary.json')
    task3_summary_path = os.path.join(output_dir, 'task3', 'evaluation_summary.json')

    if os.path.exists(task2_summary_path):
        task2_summary = read_json(task2_summary_path)
    if os.path.exists(task3_summary_path):
        task3_summary = read_json(task3_summary_path)

    if task2_summary and task3_summary:
        plot_metric_comparison(
            task2_summary,
            task3_summary,
            os.path.join(report_dir, 'metric_comparison.png')
        )

    tables = build_tables(config, split_counts, task2_summary, task3_summary)
    with open(os.path.join(report_dir, 'summary_tables.md'), 'w') as f:
        f.write(tables)

    worst_cases_path = os.path.join(output_dir, 'task3', 'worst_cases.json')
    if os.path.exists(worst_cases_path):
        worst_cases = read_json(worst_cases_path)
        with open(os.path.join(report_dir, 'worst_cases_table.md'), 'w') as f:
            f.write(build_worst_case_table(worst_cases))

    print(f'Report assets saved to {report_dir}')


if __name__ == '__main__':
    main()
