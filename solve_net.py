from utils import LOG_INFO
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[start_idx: end_idx], y[start_idx: end_idx]


def solve_net(model, dataset,
              batch_size, max_epoch, disp_freq, test_freq):

    iter_counter = 0
    loss_list = []
    accuracy_list = []
    test_acc = []
    test_loss = []
    total_loss_list = []

    for k in range(max_epoch):
        for train_sample in dataset.train_iterator(batch_size):
            iter_counter += 1
            train_input = train_sample.vec_input()
            train_label = train_sample.label()
            loss, accuracy, predicts, trues, outs = \
                model.train(train_input, train_label)
            # for i in range(batch_size):
            #     epsilon = 1e-5
            #     to_max = np.reshape(np.append(outs[i] - epsilon, np.zeros(len(outs[i]))), [2, len(outs[i])])
            #     v = np.max(to_max, axis=0) * (1 / (1 - epsilon))
            #     print("Predict %d, True %d, out %s" % (predicts[i], trues[i], str(v)))
            loss_list.append(loss)
            accuracy_list.append(accuracy)

            if iter_counter % disp_freq == 0:
                msg = 'Training iter %d, mean loss %.5f (batch loss %.5f), mean acc %.5f' % (iter_counter,
                                                                                             np.mean(loss_list),
                                                                                             loss_list[-1],
                                                                                             np.mean(accuracy_list))
                LOG_INFO(msg)
                loss_list = []
                accuracy_list = []

            if iter_counter % test_freq == 0:
                LOG_INFO('    Testing...')
                for test_sample in dataset.test_iterator(batch_size):
                    test_input = test_sample.vec_input()
                    test_label = test_sample.label()
                    t_accuracy, t_loss = model.test(test_input, test_label)
                    test_acc.append(t_accuracy)
                    test_loss.append(t_loss)

                msg = '    Testing iter %d, mean loss %.5f, mean acc %.5f' % (iter_counter,
                                                                              np.mean(test_loss),
                                                                              np.mean(test_acc))
                LOG_INFO(msg)
                test_acc = []
                test_loss = []
            if iter_counter % 100 == 0:
                total_loss_list.append(loss)
