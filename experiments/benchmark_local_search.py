"""Paired full/local support-search replay from identical saved initial supports.

Includes compaction, gradients, proposal screening and full-graph acceptance.
Excludes shared model loading, spectral preprocessing and initial saliency.
This development benchmark uses validation queries and is not a final test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gcn import load_dataset, MLPLinkPredictor
from experiments.audit_local_gcn import benchmark_pair
from src.models.gcn import GCN
from src.models.link_predictor import LinkPredictionModel
from src.explainers.candidates import candidate_edge_mask
from src.explainers.support_refinement import refine_support
from src.evaluation.interventions import evaluate_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior search benchmarks')
    if args.repeats < 2 or args.repeats % 2:
        raise ValueError('Use positive even paired repeat count')
    reference = json.loads(args.input.read_bytes())
    if reference['args']['query_split'] != 'val' or reference['args']['candidate_region'] != 'gcn-boundary':
        raise ValueError('Use boundary-protocol validation records')
    path = Path(reference['args']['checkpoint_dir']) / (reference['args']['dataset'] + '_gcn.pt')
    if sha(path) != reference['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    state = torch.load(path, map_location=args.device, weights_only=False)
    config = state['config']
    torch.manual_seed(config['seed'])
    data = load_dataset(config['dataset'])
    for name, edges in state['edge_splits'].items():
        if hashlib.sha256(edges.cpu().numpy().tobytes()).hexdigest() != reference['split_sha256'][name]:
            raise ValueError('Split hash differs')
        setattr(data, name, edges)
    data.edge_index = data.train_pos_edge_index
    data = data.to(args.device)
    encoder = GCN(config['in_channels'], config['hidden_channels'], config['out_channels'], config['num_layers'])
    predictor = MLPLinkPredictor(config['out_channels'], config['hidden_channels'])
    encoder.load_state_dict(state['model_state_dict'])
    predictor.load_state_dict(state['predictor_state_dict'])
    model = LinkPredictionModel(encoder, predictor).to(args.device).eval()
    sources = [Path(__file__), Path('experiments/train_gcn.py'), Path('experiments/audit_local_gcn.py'),
               *sorted(Path('src').rglob('*.py'))]
    hashes = {str(source): sha(source) for source in sources}
    originals = [r for r in reference['rows'] if r['method'] == 'Saliency-supportive']
    if not originals:
        raise ValueError('Missing saved initial supports')
    records, failures = [], []
    for saved in originals:
        a, b = saved['query']
        n = data.num_nodes
        mask = candidate_edge_mask(data.edge_index, n, [a, b], len(encoder.convs), 'gcn-boundary')
        edges = data.edge_index[:, mask]
        candidates = torch.unique(edges.min(0).values * n + edges.max(0).values)
        if len(candidates) != saved['candidate_edges']:
            raise ValueError('Candidate region differs')
        initial = Data(edge_index=torch.tensor(saved['support'], dtype=torch.long,
                                               device=args.device).reshape(-1, 2).t())
        before = evaluate_support(model, data, initial, a, b, args.device)
        outputs = {'full': [], 'local': []}

        def run(name):
            outputs[name].append(refine_support(model, data, a, b, candidates, initial,
                                                steps=args.steps, batch_size=args.batch_size,
                                                local_gcn=name == 'local'))

        timing = benchmark_pair(lambda: run('full'), lambda: run('local'), args.device,
                                args.repeats, warmups=1)
        results = {}
        for name, values in outputs.items():
            results[name] = []
            for support, trace in values[1:]:
                metrics = evaluate_support(model, data, support, a, b, args.device)
                valid = (metrics['support_edges'] == before['support_edges']
                         and metrics['necessity_flip'] >= before['necessity_flip']
                         and metrics['sufficiency_agreement'] >= before['sufficiency_agreement']
                         and metrics['necessity_confidence_drop'] >= before['necessity_confidence_drop'] - 1e-6
                         and max(0, metrics['sufficiency_confidence_drop']) <= max(0, before['sufficiency_confidence_drop']) + 1e-6)
                if not valid:
                    failures.append({'query': saved['query'], 'budget': saved['budget'], 'mode': name})
                results[name].append({'support': support.edge_index.t().cpu().tolist(),
                                      'metrics': metrics, 'trace': trace})
        records.append({'query': saved['query'], 'budget': saved['budget'], 'initial_metrics': before,
                        'timing': timing, 'results': results})
        print(f'Completed query={saved["query"]} budget={saved["budget"]}', flush=True)
    report = {'benchmark': 'paired-local-search-full-accept-v1',
              'input': str(args.input), 'input_sha256': sha(args.input),
              'checkpoint_sha256': sha(path), 'split_sha256': reference['split_sha256'],
              'source_sha256': hashes,
              'args': vars(args) | {'input': str(args.input), 'output': str(args.output)},
              'gpu': torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None,
              'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'scope': 'Ungrouped single-edge search from saved signed-saliency supports, same proposals/steps/acceptance rules. Timings include compaction, gradients, screening and full acceptance, but exclude model loading, saliency initialization, partition fitting, candidate-region preparation and final metric reporting. Shared-device diagnostics, not a total explanation-pipeline speed claim. Peak extra allocation is process-local and excludes persistent data/model memory.',
              'warmups_per_path': 1, 'failures': failures, 'records': records}
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    if failures:
        raise RuntimeError('Search non-regression checks failed; inspect saved output')


if __name__ == '__main__':
    main()
