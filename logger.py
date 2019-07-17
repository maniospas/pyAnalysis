import traceback


def log(*text):
    trace = "".join(traceback.format_stack())
    prefix = ""
    print(prefix+"   "*(len(traceback.format_stack())-4)+" ".join(str(param) for param in text))