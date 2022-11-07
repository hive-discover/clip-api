from api import app
from worker import run

from threading import Thread

if __name__ == '__main__':
    t = Thread(target=app.run, kwargs={'host': "0.0.0.0", 'port' : 8080}, daemon=True)
    t.start()

    run()