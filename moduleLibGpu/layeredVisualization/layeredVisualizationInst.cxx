#define VISUALIZATION_MAX_LAYERS 4

#define PRESENCE_ONCE
#define VECTOR_TYPE float16_x2
#define PRESENCE_TYPE float16
# include "layeredVisualization.inl"
#undef VECTOR_TYPE
#undef PRESENCE_TYPE
#undef PRESENCE_ONCE
