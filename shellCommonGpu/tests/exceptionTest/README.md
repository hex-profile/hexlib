# Exceptions

| OS       | Compiler | Computer Type | Fail Time     | Additional Time per Cleanup |
|----------|----------|---------------|---------------|-----------------------------|
| Windows  | MSVC     | Average       | 2100 ns       | 120 ns                      |
| Linux    | GCC      | Powerful      | 2300 ns       | 40 ns                       |

The expenses for catch (...) in case of success are almost zero for
both Windows and Linux, with the accuracy of this test.

# Error Codes

| OS       | Compiler | Computer Type | Fail Time    | Additional Time per Cleanup |
|----------|----------|---------------|--------------|-----------------------------|
| Windows  | MSVC     | Average       | 5 ns         | 3.8 ns                      |
| Linux    | GCC      | Powerful      | 3 ns         | 2 ns                        |
