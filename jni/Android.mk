LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := llama2.c
LOCAL_MODULE_FILENAME := libllama2c
LOCAL_SRC_FILES := \
    ../api.c \
	../sampler.c \
	../tokenizer.c \
	../transformer.c \
	../util.c

LOCAL_LDLIBS := -lm
LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

include $(BUILD_SHARED_LIBRARY)
