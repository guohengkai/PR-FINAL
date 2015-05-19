CC := g++
PANDORA := .
INCS := ./include /usr/local/include
CPPFLAGS := -Wall -Wunreachable-code -Werror -Wsign-compare -g -fPIC -std=c++11
LIBPATH = -L/usr/local/lib
LIBS := $(LIBPATH) -lopencv_core -lopencv_nonfree -lopencv_features2d -lopencv_ml -lopencv_imgproc -lopencv_highgui
SOURCE := $(wildcard  ./*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SOURCE))
OBJSMAIN =  $(OBJS)
%.o:%.cpp	
	$(CC) $(CPPFLAGS) $(addprefix -I,$(INCS)) -c $< -o $@
TARGETS=./hw_sign
all:$(TARGETS)
$(TARGETS) : $(OBJSMAIN)
	$(CC) $(CPPFLAGS) -o $@ $^ $(addprefix -I,$(INCS)) $(LIBS) #-Wl,-rpath,/usr/local/lib
clean:
	rm -f ./*.o
	rm -f $(TARGETS) 
test:
	$(TARGETS)
	$(TARGETS) train/face01142.jpg
	$(TARGETS) test/face.jpg
	$(TARGETS) test/nonface.jpg
