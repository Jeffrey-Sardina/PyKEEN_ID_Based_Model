from pykeen.pipeline import pipeline
from id_based_model import ID_Based_Model

def main():
    pipeline_result = pipeline(
        dataset='UMLS',
        model=ID_Based_Model,
        model_kwargs={
            'dim': 100
        },
        negative_sampler_kwargs = {
             'num_negs_per_pos': 5
        },
        training_kwargs=dict(
            num_epochs=150,
            batch_size=256
        )
    )
    mr = pipeline_result.get_metric('mr')
    mrr = pipeline_result.get_metric('mrr')
    h1 = pipeline_result.get_metric('Hits@1')
    h3 = pipeline_result.get_metric('Hits@3')
    h5 = pipeline_result.get_metric('Hits@5')
    h10 = pipeline_result.get_metric('Hits@10')
    print(f'\nMR = {mr} \nMRR = {mrr} \nHits@(1,3,5,10) = {h1, h3, h5, h10}\n')

if __name__ == '__main__':
    main()
