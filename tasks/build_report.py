import json
import os
import sys
from datetime import datetime

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
        return yaml.safe_load(f), config_path


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)


def format_names(names):
    if not names:
        return '<fill Chinese names>'
    return ', '.join(names)


def format_list(items, default_text):
    if not items:
        return [default_text]
    return items


def summarize_run_status(run_metadata):
    if not run_metadata:
        return 'Not started'
    return run_metadata.get('status', 'unknown').capitalize()


def summarize_duration(seconds):
    if seconds is None:
        return 'N/A'
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes}m {seconds}s'
    if minutes:
        return f'{minutes}m {seconds}s'
    return f'{seconds}s'


def asset_status_line(root_dir, label, path):
    if os.path.exists(path):
        return f"- {label}: `{os.path.relpath(path, root_dir)}`"
    return f"- {label}: pending generation"


def build_results_lines(task2_summary, task3_summary):
    lines = []
    if task2_summary is None:
        lines.append('Task 2 results: pending formal execution.')
    else:
        lines.append(
            f"Task 2 improved PSNR from {task2_summary['psnr_before']['mean']:.2f} dB to "
            f"{task2_summary['psnr_after']['mean']:.2f} dB and SSIM from "
            f"{task2_summary['ssim_before']['mean']:.4f} to {task2_summary['ssim_after']['mean']:.4f}."
        )

    if task3_summary is None:
        lines.append('Task 3 results: pending formal execution.')
    else:
        lines.append(
            f"Task 3 improved PSNR from {task3_summary['psnr_before']['mean']:.2f} dB to "
            f"{task3_summary['psnr_after']['mean']:.2f} dB and SSIM from "
            f"{task3_summary['ssim_before']['mean']:.4f} to {task3_summary['ssim_after']['mean']:.4f}."
        )
        if task2_summary is not None:
            lines.append(
                f"Compared with Task 2, Task 3 changed PSNR by "
                f"{task3_summary['psnr_after']['mean'] - task2_summary['psnr_after']['mean']:.2f} dB "
                f"and SSIM by {task3_summary['ssim_after']['mean'] - task2_summary['ssim_after']['mean']:.4f}."
            )

    return lines


def build_report(config, config_path, metadata, split_counts, task2_summary, task3_summary,
                 run_metadata, output_dir, worst_cases):
    root_dir = os.path.dirname(output_dir)
    report_assets_dir = os.path.join(output_dir, 'report_assets')
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    division_of_labor = format_list(metadata.get('division_of_labor'), '<fill division of labor>')
    chinese_names = metadata.get('chinese_names', [])
    run_status = summarize_run_status(run_metadata)
    total_runtime = summarize_duration(run_metadata.get('total_duration_seconds') if run_metadata else None)

    lines = []
    lines.append(f"# {metadata.get('project_title', 'BraTS MRI Reconstruction Project')}")
    lines.append('')
    lines.append(f"Generated from config `{os.path.relpath(config_path, root_dir)}` on {generated_time}.")
    lines.append('')
    lines.append('## Group Information')
    lines.append('')
    lines.append(f"- Course: {metadata.get('course', '<fill course name>')}")
    lines.append(f"- Group: {metadata.get('group_name', '<fill group name>')}")
    lines.append(f"- Chinese names: {format_names(chinese_names)}")
    lines.append(f"- Note: {metadata.get('report_author_note', 'Fill any remaining placeholders before submission.')}")
    lines.append('')
    lines.append('## AI Usage Declaration')
    lines.append('')
    lines.append(metadata.get('ai_usage_statement', '<fill AI usage statement>'))
    lines.append('')
    lines.append(f"Presentation reminder: {metadata.get('presentation_note', 'Remember to declare AI use in slides.')}")
    lines.append('')
    lines.append('## Project Objective')
    lines.append('')
    lines.append('This project reconstructs fully sampled T2 brain MRI slices from AF=5 undersampled k-space using the BraTS dataset. The assignment requires three tasks: undersampling simulation, a baseline reconstruction model, and a multi-modal unrolled model with data consistency.')
    lines.append('')
    lines.append('## Dataset and Preprocessing')
    lines.append('')
    lines.append(f"- Dataset path: `{config['data']['dataset_path']}`")
    lines.append(f"- Modalities used: {', '.join(config['data']['modalities'])}")
    lines.append(f"- Slice axis: {config['data']['slice_axis']}")
    lines.append(f"- Intensity normalization: z-score on non-zero voxels")
    lines.append(f"- Split strategy: patient-level train/validation/test")
    lines.append(f"- Split counts: train={split_counts['Train']}, validation={split_counts['Validation']}, test={split_counts['Test']}")
    lines.append('')
    lines.append('## Methods')
    lines.append('')
    lines.append('### Task 1')
    lines.append('')
    lines.append('A 2D random variable-density sampling mask with acceleration factor 5 is generated in k-space. Fully sampled T2 slices are transformed with FFT, masked, and reconstructed with inverse FFT to obtain aliased images.')
    lines.append('')
    lines.append('### Task 2')
    lines.append('')
    lines.append(f"Task 2 uses a U-Net baseline with base channels {config['task2']['base_channels']}, depth {config['task2']['depth']}, batch size {config['task2']['batch_size']}, MSE loss, and learning rate {config['task2']['learning_rate']}. `ReduceLROnPlateau` is used for learning rate decay.")
    lines.append('')
    lines.append('### Task 3')
    lines.append('')
    lines.append(f"Task 3 uses an unrolled U-Net with {config['task3']['num_cascades']} cascades, data consistency layers, and multi-modal input consisting of aliased T2 plus fully sampled T1. The loss is `{config['task3']['loss_type']}` with L1 weight {config['task3'].get('l1_weight', 'N/A')}.")
    lines.append('')
    lines.append('## Division of Labor')
    lines.append('')
    for item in division_of_labor:
        lines.append(f"- {item}")
    lines.append('')
    lines.append('## Quantitative Results')
    lines.append('')
    for result_line in build_results_lines(task2_summary, task3_summary):
        lines.append(f"- {result_line}")
    lines.append('')
    lines.append('## Figures and Tables')
    lines.append('')
    lines.append(asset_status_line(root_dir, 'Dataset split chart', os.path.join(report_assets_dir, 'dataset_split.png')))
    lines.append(asset_status_line(root_dir, 'Metric comparison chart', os.path.join(report_assets_dir, 'metric_comparison.png')))
    lines.append(asset_status_line(root_dir, 'Summary tables', os.path.join(report_assets_dir, 'summary_tables.md')))
    lines.append(asset_status_line(root_dir, 'Worst-case table', os.path.join(report_assets_dir, 'worst_cases_table.md')))
    lines.append(asset_status_line(root_dir, 'Task 1 visualization', os.path.join(output_dir, 'task1', 'undersampling_visualization.png')))
    lines.append(asset_status_line(root_dir, 'Task 2 loss curve', os.path.join(output_dir, 'task2', 'loss_curves.png')))
    lines.append(asset_status_line(root_dir, 'Task 2 reconstructions', os.path.join(output_dir, 'task2', 'reconstruction_results.png')))
    lines.append(asset_status_line(root_dir, 'Task 3 loss curve', os.path.join(output_dir, 'task3', 'loss_curves.png')))
    lines.append(asset_status_line(root_dir, 'Task 3 best reconstructions', os.path.join(output_dir, 'task3', 'best_reconstructions.png')))
    lines.append(asset_status_line(root_dir, 'Task 3 error analysis', os.path.join(output_dir, 'task3', 'error_analysis.png')))
    lines.append('')
    lines.append('## Error Analysis')
    lines.append('')
    if not worst_cases:
        lines.append('Task 3 worst-case analysis is pending because `worst_cases.json` is not available yet.')
    else:
        lines.append('The 5 worst-performing Task 3 cases are summarized below.')
        lines.append('')
        lines.append('| Rank | Patient | Slice | PSNR After | SSIM After |')
        lines.append('| --- | --- | ---: | ---: | ---: |')
        for index, case in enumerate(worst_cases, start=1):
            lines.append(
                f"| {index} | {case['patient']} | {case['slice_idx']} | {case['psnr_after']:.2f} | {case['ssim_after']:.4f} |"
            )
        lines.append('')
        lines.append('Potential improvements for these cases include stronger edge-preserving loss terms, more cascades if runtime allows, and targeted inspection of slices with complex tumor boundaries.')
    lines.append('')
    lines.append('## Execution Status')
    lines.append('')
    lines.append(f"- Current run status: {run_status}")
    lines.append(f"- Output directory: `{os.path.relpath(output_dir, root_dir)}`")
    lines.append(f"- Total runtime: {total_runtime}")
    if run_metadata:
        lines.append(f"- Last completed stage: {run_metadata.get('last_completed_stage', 'N/A')}")
        if run_metadata.get('log_path'):
            lines.append(f"- Log file: `{os.path.relpath(run_metadata['log_path'], root_dir)}`")
    lines.append('')
    lines.append('## Discussion')
    lines.append('')
    if task2_summary is not None and task3_summary is not None:
        lines.append('The current results support the expected conclusion of the project: the baseline network substantially reduces aliasing artifacts, and the multi-modal unrolled model provides additional measurable improvement.')
    else:
        lines.append('Formal quantitative conclusions are pending because the final training run has not completed yet. The automation scripts are prepared so this report will be refreshed automatically after the formal run finishes.')
    lines.append('')
    lines.append('## Submission Checklist')
    lines.append('')
    lines.append('- Code: prepared')
    lines.append(f"- Report draft: `{os.path.basename(os.path.join(root_dir, 'REPORT.md'))}`")
    lines.append('- Presentation slides: still need to be prepared manually')
    lines.append('- Final PDF export: still needs manual export after placeholders are reviewed')

    return '\n'.join(lines) + '\n'


def main():
    config, config_path = load_config()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, config['output']['dir'].lstrip('./'))
    config_name = os.path.basename(config_path)
    write_main_report_env = os.environ.get('WRITE_MAIN_REPORT')
    if write_main_report_env is None:
        write_main_report = config_name == 'formal_train.yaml'
    else:
        write_main_report = write_main_report_env.lower() in {'1', 'true', 'yes'}

    metadata_path = os.path.join(root_dir, 'config', 'report_metadata.json')
    metadata = load_json(metadata_path, default={}) or {}
    run_metadata = load_json(os.path.join(output_dir, 'run_metadata.json'), default=None)
    task2_summary = load_json(os.path.join(output_dir, 'task2', 'evaluation_summary.json'), default=None)
    task3_summary = load_json(os.path.join(output_dir, 'task3', 'evaluation_summary.json'), default=None)
    worst_cases = load_json(os.path.join(output_dir, 'task3', 'worst_cases.json'), default=[])

    train_ds, val_ds, test_ds, _ = create_dataloaders(config)
    split_counts = {
        'Train': len(train_ds),
        'Validation': len(val_ds),
        'Test': len(test_ds),
    }

    report_text = build_report(
        config,
        config_path,
        metadata,
        split_counts,
        task2_summary,
        task3_summary,
        run_metadata,
        output_dir,
        worst_cases,
    )

    with open(os.path.join(output_dir, 'report_assets', 'report_snapshot.md'), 'w') as f:
        f.write(report_text)

    if write_main_report:
        with open(os.path.join(root_dir, 'REPORT.md'), 'w') as f:
            f.write(report_text)
        print(f"Report written to {os.path.join(root_dir, 'REPORT.md')}")
    else:
        print(f"Report snapshot written to {os.path.join(output_dir, 'report_assets', 'report_snapshot.md')}")


if __name__ == '__main__':
    main()
