## Face Controller

This project was created in the scope of an university assignment where the
goal was to create an application that would allow the user to control their
computer in a unique way. We chose face and gesture recognition as our input
medium and wrote a simple UI to map the keys to the preconfigured actions.

## How to run

Clone the repository and run:

```sh
pip install .
python src/main.py
```

If you want to have debug info printed on the webcam output and the terminal
use the `--debug` or `-d` flag.

You can change the mouse sensitivity using the `-s` flag. (e.g. `-s 20`)

This project was originally written using python version 3.10.14 but should
work on newer python versions as well.
