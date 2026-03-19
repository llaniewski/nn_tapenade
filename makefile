CFLAGS=-g

main : main.o nn.o nn_b.o
	g++ $(CFLAGS) -o $@ $^ -lnlopt

%.o : %.c const.h
	gcc $(CFLAGS) -c $<

%.o : %.cpp const.h
	g++ $(CFLAGS) -c $<

%_b.c : %.c const.h
	tapenade -root "NeuralNetworkLoss(loss)/(weights)" -b $<
	sed -e '/adStack/s|^|//|' -i $@

.PRECIOUS: %_b.c
