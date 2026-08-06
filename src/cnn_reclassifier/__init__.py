"""
Etapa 2 del framework "Where's Wally?" (Barthelmes & Vidal, 2021).

Este paquete implementa la CNN ligera que RE-CLASIFICA los candidatos
propuestos por el Haar-cascade classifier (etapa 1). El cascade es bueno
diciendo donde NO esta Wally (descarta el fondo) pero genera muchos falsos
positivos; la CNN se queda solo con los que son realmente Wally (prob > 90%).
"""
