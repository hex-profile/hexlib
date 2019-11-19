
    //----------------------------------------------------------------
    //
    // Testing
    //
    //----------------------------------------------------------------

    if (0)
    {
        for (int t = 0; t < 10000; ++t)
        {
            vector<StlString> testArgs;

            ////

            int nargs = randRange() % 20;

            for (int k = 0; k < nargs; ++k)
            {
                int arglen = randRange() % 6;

                StlString s;

                for (int i = 0; i < arglen; ++i)
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

            for (size_t i = 0; i < testArgs.size(); ++i)
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
