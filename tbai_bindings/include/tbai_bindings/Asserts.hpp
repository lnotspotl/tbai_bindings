#ifdef TBAI_BINDINGS_DISABLE_ASSERTS
#define TBAI_BINDINGS_ASSERT(condition, message)
#else

#define TBAI_BINDINGS_ASSERT(condition, message)                                                                  \
    do {                                                                                                          \
        if (!(condition)) {                                                                                       \
            std::cerr << "\n"                                                                                     \
                      << "Assertion failed: " << #condition << " in file " << __FILE__ << " at line " << __LINE__ \
                      << ": " << message << "\n"                                                                  \
                      << std::endl;                                                                               \
            std::abort();                                                                                         \
        }                                                                                                         \
    } while (0)

#endif