from flask import Flask, url_for
from network_setup import network_setup
import realtime
import play
import json
from threading import Thread

app = Flask(__name__)
json_result = None


def api_server_start():
    app.run(host='localhost', port='8080')


@app.route('/')
def api_root():
    return 'Welcome'


@app.route('/notes.json')
def api_notes():
    return json.dumps(json_result, separators=(',', ':'))

if __name__ == '__main__':
    model_file_path = 'models/ucho-2048-big-batch.bin'
    model, dataset, freq_count, count_bins = network_setup(model_file_path)
    duration = 0.5

    thread = Thread(target=api_server_start)
    thread.start()

    for (t, data) in realtime.streaming():
        result = play.streaming(model, data,
                                time=t,
                                freq_count=freq_count,
                                count_bins=count_bins,
                                duration=duration)
        json_result = {'time': result['time'],
                       'activated': result['activated']}
