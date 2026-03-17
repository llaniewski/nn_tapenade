CFLAGS=-g

main : main.o nn.o nn_b.o
	g++ $(CFLAGS) -o $@ $^

%.o : %.c
	gcc $(CFLAGS) -c $<

%.o : %.cpp
	g++ $(CFLAGS) -c $<

%_b.c : %.c
	tapenade -root 'NeuralNetwork()/(weights)' -b $<
	sed -e '/adStack/s|^|//|' -i $@

.PRECIOUS: %_b.c
