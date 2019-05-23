import subprocess

runs = [
    'python main.py --model_name="TreeLSTM_RepSize_300_fastText_Batch64_V2" --model=treeLSTM --word_min_count=50 --word_embed_model=fasttext --word_embed_mode=pretrained --lr_decay=1 --learning_rate=0.1 --dropout_prob=0.1 --epochs=0 --pretrain_max_epoch=200 --conv_cond=100 --load_model=True --batch_size=64 --sentence_embedding_size=300 --use_selective_training=False --use_gpu=True ',
    'python main.py --model_name="TreeLSTM_RepSize_300_fastText_Batch64_V3" --model=treeLSTM --word_min_count=50 --word_embed_model=fasttext --word_embed_mode=pretrained --lr_decay=1 --learning_rate=0.1 --dropout_prob=0.1 --epochs=0 --pretrain_max_epoch=200 --conv_cond=100 --load_model=True --batch_size=64 --sentence_embedding_size=300 --use_selective_training=False --use_gpu=True '
]

for run in runs:
    subprocess.run(run, shell=True)
    for i in range(10):
        print('NEXT MODEL!\n')

print('NO MORE MODELS..... :-(')
