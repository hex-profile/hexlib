
    //----------------------------------------------------------------
    //
    // Testing
    //
    //----------------------------------------------------------------

    if (0)
    {
        for_count_ex (t, 10000)
        {
            vector<StlString> testArgs;

            ////

            int nargs = randRange() % 20;

            for_count (k, nargs)
            {
                int arglen = randRange() % 6;

                StlString s;

                for_count (i, arglen)
                {
                    int r = randRange() % 4;

                    if (r == 0)
                        s += CHARTEXT('"');
                    else if (r == 1)
                        s += CHARTEXT('\\');
                    else if (r == 2)
                        s += CHARTEXT(' ');
                    else if (r == 3)
                        s.append(1, CharType(randRange() % 256));
                }

                testArgs.push_back(s);
            }

            ////

            StlString cmdline;

            for_count (i, testArgs.size())
            {
                cmdLine::convertToArg(testArgs[i], cmdline);
                cmdline += CHARTEXT(" ");
            }

            ////

            vector<StlString> reparseArg;
            cmdLine::parseCmdLine(cmdline, reparseArg);

            REQUIRE(reparseArg == testArgs);
        }

        printMsg(kit.log, CT("testOk"));
    }
