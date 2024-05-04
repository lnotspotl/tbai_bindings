#pragma once

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

#ifdef TBAI_BINDINGS_DISABLE_PRINTS
#define TBAI_BINDINGS_PRINT(message)
#else
#define TBAI_BINDINGS_PRINT(message) std::cout << "[Tbai bindings] | " << message << std::endl
#endif

#define TBAI_BINDINGS_STD_THROW(message)                                                                                    \
    do {                                                                                                           \
        std::cerr << "\n"                                                                                          \
                  << "Exception thrown in file " << __FILE__ << " at line " << __LINE__ << ": " << message << "\n" \
                  << std::endl;                                                                                    \
        throw std::runtime_error(message);                                                                         \
    } while (0)

#define TBAI_BINDINGS_STD_THROW_IF(condition, message) \
    if (condition) {                          \
        TBAI_BINDINGS_STD_THROW(message);              \
    }