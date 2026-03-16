
main : main.o nn.o nn_b.o
	g++ -o $@ $^

%.o : %.c
	gcc -c $<

%.o : %.cpp
	g++ -c $<

%_b.c : %.c
	tapenade -root 'NeuralNetwork()/(weights)' -b $<
	sed -e '/adStack/s|^|//|' -i $@
