from solve_net import solve_net
from play import play, streaming
import realtime
import signal
import sys
from network_setup import network_setup


model_file_path = None
play_file_name = None
start_time = 0.5

# Check parameters
if len(sys.argv) >= 2:
    if sys.argv[1] == 'train':
        train_or_play = True
    else:
        train_or_play = False
        model_file_path = sys.argv[2]
        play_file_name = sys.argv[3]
        if len(sys.argv) >= 5:
            start_time = float(sys.argv[4])
else:
    raise RuntimeError("Wrong argument")

model, dataset, freq_count, count_bins = network_setup(model_file_path)
duration = 0.5

if train_or_play:

    # Initialize signal handler to save model at SIGINT
    def signal_handler(sig, frame):
        print('Save model? [y/n]')
        save_or_not = raw_input()
        if save_or_not == 'y':
            model.dumps('model.bin')
        print('Exiting...')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    solve_net(model, dataset,
              max_epoch=50, disp_freq=200, test_freq=2000)

    print('Save model? [y/n]')
    yes_or_no = raw_input()
    if yes_or_no == 'y':
        model.dumps('model.bin')
else:
    # while True:
    #     result = play(model, play_file_name,
    #                   start_time=start_time,
    #                   freq_count=freq_count,
    #                   count_bins=count_bins,
    #                   duration=duration)
    #     print(result)
    #     print('Start time(seconds, minus value for exit):')
    #     user_input = raw_input()
    #     if len(user_input) == 0 or float(user_input) < 0:
    #         sys.exit(0)
    #     start_time = float(user_input)
    for (t, data) in realtime.streaming():
        streaming(model, data,
                  time=t,
                  freq_count=freq_count,
                  count_bins=count_bins,
                  duration=duration)
        # print(result)

