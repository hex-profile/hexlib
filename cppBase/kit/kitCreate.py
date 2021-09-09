#!/usr/bin/env python

#================================================================
#
# writeLn
#
#================================================================

def writeLn(file, level, text):
    file.write('    ' * level[0] + text + '\n')

#================================================================
#
# Indent
#
#================================================================

class Indent(object):

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        self.level[0] += 1

    def __exit__(self, type, value, traceback):
        self.level[0] -= 1

#================================================================
#
# main
#
#================================================================

if __name__ == '__main__':

    #----------------------------------------------------------------
    #
    # Save
    #
    #----------------------------------------------------------------

    with open('kitCreate.inl', 'wt') as file:

        level = [0]

        for n in range(2, 16 + 1):

            write = lambda s : writeLn(file, level, s)
            writes = lambda s : writeLn(file, level, s + (' ' if len(s) else '') + '\\')

            write('//----------------------------------------------------------------')
            write('')
            writes('#define KIT__CREATE%d(Kit, %s)' % (n, ", ".join(['Type%d, name%d' % (i, i) for i in range(0, n)])))

            with Indent(level):

                writes('')
                writes('struct Kit')

                with Indent(level):
                    writes(':')
                    [writes('Kit_FieldTag<struct name%d##_Tag>%s' % (i, '' if i == n-1 else ',')) for i in range(n)]

                writes('{')

                with Indent(level):

                    #----------------------------------------------------------------
                    #
                    # Fields.
                    #
                    #----------------------------------------------------------------

                    writes('')
                    [writes('Type%d name%d;' % (i, i)) for i in range(n)]
                    writes('')

                    #----------------------------------------------------------------
                    #
                    # By params.
                    #
                    #----------------------------------------------------------------

                    writes('sysinline Kit')
                    writes('(')
                    with Indent(level):
                        [writes('ParamType<Type%d>::T name%d%s' % (i, i, '' if i == n-1 and n != 1 else ',')) for i in range(n)]
                    writes(')')

                    with Indent(level):
                        writes(':')
                        [writes('name%d(name%d)%s' % (i, i, '' if i == n-1 else ',')) for i in range(n)]

                    writes('{')
                    writes('}')
                    writes('')

                    #----------------------------------------------------------------
                    #
                    # By any other kit.
                    #
                    #----------------------------------------------------------------

                    writes('template <typename OtherKit>')
                    writes('sysinline Kit(const OtherKit& otherKit)')

                    with Indent(level):
                        writes(':')
                        [writes('name%d(otherKit.name%d)%s' % (i, i, '' if i == n-1 else ',')) for i in range(n)]

                    writes('{')
                    writes('}')
                    writes('')

                    #----------------------------------------------------------------
                    #
                    # Replacing constructor.
                    #
                    #----------------------------------------------------------------

                    writes('template <typename OldKit, typename NewKit>')
                    writes('sysinline Kit(const OldKit& oldKit, Kit_ReplaceConstructor, const NewKit& newKit)')

                    with Indent(level):
                        writes(':')
                        [writes('name%d(Kit_Replacer<OldKit, NewKit, name%d##_Tag>::func(&oldKit, &newKit)->name%d)%s' % (i, i, i, '' if i == n-1 else ',')) for i in range(n)]

                    writes('{')
                    writes('}')

                write('}')
                write('')

            #----------------------------------------------------------------
            #
            #
            #
            #----------------------------------------------------------------

            writes('#define KIT_CREATE%d(Kit, %s)' % (n, ", ".join(['Type%d, name%d' % (i, i) for i in range(0, n)])))
            with Indent(level):
                write('KIT__CREATE%d(Kit, %s)' % (n, ", ".join(['Type%d, name%d' % (i, i) for i in range(0, n)])))

            write('')

            write('')
