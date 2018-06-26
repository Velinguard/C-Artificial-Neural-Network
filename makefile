LDLAGS  = -lm

network: train.o ann.o layer.o
	$(CC) obj/train.o obj/ann.o obj/layer.o -o network $(LDLAGS)

train.o:
	$(CC) testsuite/train.c -c -o obj/train.o

ann.o:
	$(CC) src/ann.c -c -o obj/ann.o

layer.o:
	$(CC) src/layer.c -c -o obj/layer.o


clean:
	rm -f sort train.o ann.o layer.o

.PHONY: clean