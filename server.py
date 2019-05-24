from flask import Flask
import os
import socket
import logging

import check_sents


app = Flask(__name__)


@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"

    return html.format(name=os.getenv("NAME", "world"),
                       hostname=socket.gethostname())


@app.route("/fix-sent/<sent>")
def fix_sent(sent):
    return check_sents.fix_sentence(sent)


def main():
    print('Server is running')
    deps_file = os.environ.get('deps', '')
    if not deps_file:
        logging.error('No deps file. Set deps env in docker-compose.yml')
        return 1

    check_sents.set_deps(deps_file)

    app.run(host='0.0.0.0', port=8000)


if __name__ == "__main__":
    main()
