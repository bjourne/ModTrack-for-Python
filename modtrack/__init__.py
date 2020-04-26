class DebugPrint:
    def __init__(self, enabled):
        self.indent = 0
        self.enabled = enabled

    def print_indented(self, text):
        if self.enabled:
            print(' ' * self.indent + text)

    def header(self, name, fmt, args):
        self.print_indented('* %s %s' % (name, fmt % args))
        self.indent += 2

    def print(self, fmt, args):
        self.print_indented(fmt % args)

    def leave(self):
        self.indent -= 2

DP = DebugPrint(True)
