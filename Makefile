CC := g++
PANDORA := .
INCS := ./thirdparty/libsvm /usr/local/include
CPPFLAGS := -Wall -Wunreachable-code -Werror -Wsign-compare -g -fPIC -std=c++11
LIBDIR = ./lib
LIBPATH = -L/usr/local/lib
LIBS := $(LIBPATH) -lopencv_core -lopencv_nonfree -lopencv_ocl -lopencv_features2d -lopencv_ml -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_gpu -lopencv_objdetect $(LIBDIR)/libsvm.so.2 $(LIBDIR)/libvl.so
SOURCE := $(wildcard  ./*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(SOURCE))
OBJSMAIN =  $(OBJS)
%.o:%.cpp	
	$(CC) $(CPPFLAGS) $(addprefix -I,$(INCS)) -c $< -o $@
TARGETS=./hw_sign
all:$(TARGETS)
$(TARGETS) : $(OBJSMAIN)
	$(CC) $(CPPFLAGS) -o $@ $^ $(addprefix -I,$(INCS)) $(LIBS) -Wl,-rpath,$(LIBDIR)
clean:
	rm -f ./*.o
	rm -f $(TARGETS) 
test:
	$(TARGETS)
