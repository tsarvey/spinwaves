"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""

from pretty_symbology import hobj, vobj, xsym, pretty_use_unicode

class stringPict(object):
    """A ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    #special value for stringPict.below
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        #picture is a string that just can be printed
        self.picture = stringPict.equalLengths(s.splitlines())
        #baseline is the line number of the "base line"
        self.baseline = baseline
        self.binding = None

    def __len__(self):
        return len(str(self))

    @staticmethod
    def equalLengths(lines):
        # empty lines
        if not lines:
            return ['']

        width = max(len(line) for line in lines)
        return [line.center(width) for line in lines]

    def height(self):
        return len(self.picture)

    def width(self):
        return len(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        #convert everything to stringPicts
        objects = []
        for arg in args:
            if isinstance(arg, basestring): arg = stringPict(arg)
            objects.append(arg)

        #make a list of pictures, with equal height and baseline
        newBaseline = max(obj.baseline for obj in objects)
        newHeightBelowBaseline = max(
            obj.height()-obj.baseline
            for obj in objects)
        newHeight = newBaseline + newHeightBelowBaseline

        pictures = []
        for obj in objects:
            oneEmptyLine = [' '*obj.width()]
            basePadding = newBaseline-obj.baseline
            totalPadding = newHeight-obj.height()
            pictures.append(
                oneEmptyLine * basePadding +
                obj.picture +
                oneEmptyLine * (totalPadding-basePadding))

        result = [''.join(lines) for lines in zip(*pictures)]
        return '\n'.join(result), newBaseline

    def right(self, *args):
        """Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.
        >>> print stringPict("10").right(" + ",stringPict("1\r-\r2",1))[0]
             1
        10 + -
             2
        """
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        return stringPict.next(*(args+(self,)))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        #convert everything to stringPicts; keep LINE
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, basestring):
                arg = stringPict(arg)
            objects.append(arg)

        #compute new width
        newWidth = max(
            obj.width()
            for obj in objects
            if obj is not stringPict.LINE)

        lineObj = stringPict(hobj('-', newWidth))

        #replace LINE with proper lines
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj

        #stack the pictures, and center the result
        newPicture = []
        for obj in objects:
            newPicture.extend(obj.picture)
        newPicture = [line.center(newWidth) for line in newPicture]
        newBaseline = objects[0].height()+objects[1].baseline
        return '\n'.join(newPicture), newBaseline

    def below(self, *args):
         """Put pictures under this picture.
         Returns string, baseline arguments for stringPict.
         Baseline is baseline of top picture
         >>> print stringPict("x+3").below(stringPict.LINE, '3')[0] #doctest: +NORMALIZE_WHITESPACE
         x+3
         ---
          3
         """
         s, baseline = stringPict.stack(self, *args)
         return s, self.baseline

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        string, baseline = stringPict.stack(*(args+(self,)))
        baseline = len(string.splitlines())-self.height()+self.baseline
        return string, baseline

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline

        # XXX this is a hack -- ascii parens are ugly!
        if ifascii_nougly and not pretty_use_unicode():
            h = 1
            b = 0

        res = self

        if left:
            lparen = stringPict(vobj(left,  h), baseline=b)
            res    = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res    = stringPict(*res.right(rparen))

        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        # XXX not used anywhere ?
        height = max(
            self.baseline,
            self.height()-1-self.baseline)*2 + 1
        slash = '\n'.join(
            ' '*(height-i-1)+xobj('/',1)+' '*i
            for i in range(height)
            )
        return self.left(stringPict(slash, height//2))

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        # XXX not used anywhere
        # XXX duplicate of root drawing in pretty.py
        #put line over expression
        result = self.above('_'*self.width())
        #construct right half of root symbol
        height = self.height()
        slash = '\n'.join(
            ' ' * (height-i-1) + '/' + ' ' * i
            for i in range(height)
            )
        slash = stringPict(slash, height-1)
        #left half of root symbol
        if height > 2:
            downline = stringPict('\\ \n \\',1)
        else:
            downline = stringPict('\\')
        #put n on top, as low as possible
        if n is not None and n.width()>downline.width():
            downline = downline.left(' '*(n.width()-downline.width()))
            downline = downline.above(n)
        #build root symbol
        root = downline.right(slash)
        #glue it on at the proper height
        #normally, the root symbel is as high as self
        #which is one less than result
        #this moves the root symbol one down
        #if the root became higher, the baseline has to grow too
        root.baseline = result.baseline-result.height()+root.height()
        return result.left(root)

    def terminal_string(self):
        """Return the string form of self such that it can be printed
           on the terminal without being broken up.
        """

        # Attempt to get a terminal width
        ncols = self.terminal_width()
        ncols -= 2
        if ncols <= 0:
            ncols = 78

        # If smaller than the terminal width, no need to correct
        if self.width() <= ncols:
            return type(self.picture[0])(self)

        # for one-line pictures we don't need v-spacers. on the other hand, for
        # multiline-pictures, we need v-spacers between blocks, compare:
        #
        #    2  2        3    | a*c*e + a*c*f + a*d  | a*c*e + a*c*f + a*d  | 3.14159265358979323
        # 6*x *y  + 4*x*y  +  |                      | *e + a*d*f + b*c*e   | 84626433832795
        #                     | *e + a*d*f + b*c*e   | + b*c*f + b*d*e + b  |
        #      3    4    4    |                      | *d*f                 |
        # 4*y*x  + x  + y     | + b*c*f + b*d*e + b  |                      |
        #                     |                      |                      |
        #                     | *d*f
        do_vspacers = (self.height() > 1)

        i = 0
        svals = []
        while i < self.width():
            svals.extend([ sval[i:i+ncols] for sval in self.picture ])
            if (self.height() > 1):
                svals.append("") # a vertical spacer
            i += ncols

        if svals[-1] == '':
            del svals[-1] #  Get rid of the last spacer

        _str = type(self.picture[0])
        return _str.join(_str("\n"), svals)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        ncols = 0
        try:
            import curses
            try:
                curses.setupterm()
                ncols = curses.tigetnum('cols')
            except AttributeError:
                # windows curses doesn't implement setupterm or tigetnum
                # code below from
                # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440694
                from ctypes import windll, create_string_buffer
                # stdin handle is -10
                # stdout handle is -11
                # stderr handle is -12
                h = windll.kernel32.GetStdHandle(-12)
                csbi = create_string_buffer(22)
                res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
                if res:
                    import struct
                    (bufx, bufy, curx, cury, wattr,
                     left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
                    ncols = right - left + 1
            except curses.error:
                pass
        except (ImportError, TypeError):
            pass
        return ncols

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstace(o, stringPict):
            return o.picture == self.picture
        return False

    def __str__(self):
        return str.join('\n', self.picture)

    def __unicode__(self):
        return unicode.join(u'\n', self.picture)

    def __repr__(self):
        return "stringPict(%r,%d)"%('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        return self.picture[index]

    def __len__(self):
        return len(self.s)

class prettyForm(stringPict):
    """Extension of the stringPict class that knows about
    basic math applications, optimizing double minus signs.
    "Binding" is interpreted as follows:
    ATOM this is an atom: never needs to be parenthesised
    FUNC this is a function application: parenthesise if added (?)
    DIV  this is a division: make wider division if divided
    POW  this is a power: only parenthesise if exponent
    MUL  this is a multiplication: parenthesise if powered
    ADD  this is an addition: parenthesise if multiplied or powered
    NEG  this is a negative number: optimise if added, parenthesise if multiplied or powered
    """
    ATOM, FUNC, DIV, POW, MUL, ADD, NEG = range(7)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """Initialize from stringPict and binding power."""
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        self.unicode = unicode or s

    def __add__(self, *others):
        """Make a pretty addition.
        Addition of negative numbers is simplified.
        """
        arg = self
        if arg.binding > prettyForm.NEG: arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            #add parentheses for weak binders
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            #use existing minus sign if available
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(binding=prettyForm.ADD, *stringPict.next(*result))

    def __div__(self, den, slashed = False):
        """Make a pretty division; stacked or slashed.
        """
        if slashed: raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding==prettyForm.DIV: num = stringPict(*num.parens())
        if den.binding==prettyForm.DIV: den = stringPict(*den.parens())

        return prettyForm(binding=prettyForm.DIV, *stringPict.stack(
            num,
            stringPict.LINE,
            den))

    def __truediv__(self, o):
        return self.__div__(o)

    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        args = self
        if args.binding > prettyForm.MUL: arg = stringPict(*args.parens())
        result = [args]
        for arg in others:
            result.append(xsym('*'))
            #add parentheses for weak binders
            if arg.binding > prettyForm.MUL: arg = stringPict(*arg.parens())
            result.append(arg)
        len_res = len(result)
        for i in xrange(len_res):
            if i < len_res-1 and result[i] == '-1' and result[i+1] == xsym('*'):
                # substitute -1 by -, like in -1*x -> -x
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            # if there is a - sign in front of all
            bin = prettyForm.NEG
        else:
            bin = prettyForm.MUL
        return prettyForm(binding=bin, *stringPict.next(*result))

    def __repr__(self):
        return "prettyForm(%r,%d,%d)"%(
            '\n'.join(self.picture),
            self.baseline,
            self.binding)

    def __pow__(self, b):
        """Make a pretty power.
        """
        a = self
        if a.binding > prettyForm.FUNC: a = stringPict(*a.parens())
        if b.binding == prettyForm.POW: b = stringPict(*b.parens())

        if a.binding == prettyForm.FUNC:
            #     2     <-- top
            #  sin (x)  <-- bot
            top = stringPict(*b.left(' '*a.prettyFunc.width()))
            top = stringPict(*top.right(' '*a.prettyArgs.width()))
            bot = stringPict(*a.prettyFunc.right(' '*b.width()))
            bot = stringPict(*bot.right(a.prettyArgs))
        else:
            #      2    <-- top
            # (x+y)     <-- bot
            top = stringPict(*b.left(' '*a.width()))
            bot = stringPict(*a.right(' '*b.width()))

        return prettyForm(binding=prettyForm.POW, *bot.above(top))

    simpleFunctions = ["sin", "cos", "tan"]
    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        if function in prettyForm.simpleFunctions:
            #simple function: use only space if possible
            assert len(args)==1, "Simple function %s must have 1 argument"%function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                #optimization: no parentheses necessary
                return prettyForm(binding=prettyForm.FUNC, *arg.left(function+' '))
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(binding=prettyForm.ATOM, *argumentList.left(function))
