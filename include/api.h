#ifndef __API_H__
#define __API_H__

#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

#endif // __API_H__
