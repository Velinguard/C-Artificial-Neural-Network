LDLAGS  = -lm

network: train.o ann.o layer.o
	$(CC) train.o ann.o layer.o -o network $(LDLAGS)

train.o:
	$(CC) train.c -c -o train.o

ann.o:
	$(CC) ann.c -c -o ann.o

layer.o:
	$(CC) layer.c -c -o layer.o


clean:
	rm -f sort train.o ann.o layer.o

.PHONY: clean