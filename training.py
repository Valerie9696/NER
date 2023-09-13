import sys
import lstm_baseline as lb
import pretrained_transformer as pt
import bert_crf as bc
from torch import cuda


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    arguments = sys.argv
    run_count = int(arguments[1])
    model = arguments[2]
    f1 = 0
    for i in range(0, run_count):
        if model == 'LSTM':
            lstm_base = lb.LSTM_Base(run_training=True)
        elif model == 'Bert':
            bert_base = pt.BertBase(filter_dataset=True)
            f1 = f1 + bert_base.final_f1
        elif model == 'BertCrf':
            bert_crf = bc.BertExtended(name='bert_crf', filter_dataset=True, with_features=True)
            f1 = f1 + bert_crf.final_f1
        elif model == 'BertLSTMCrf':
            bc.BertExtended(name='bert_lstm_crf', filter_dataset=False)

    print('Final score ', f1/run_count)
