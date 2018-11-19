from __future__ import print_function
from random import randint, choice, random, uniform

class Grammar:
    def __init__(self, path):
        self.grammar = self.get_grammar(path)

    def read_grammar(self, path):
        with open(path, 'r') as f_in:
            raw_grammar = f_in.readlines()
            return raw_grammar
        return None


    def parse_grammar(self, raw_grammar):
        grammar = {}
        start_symbol = None

        for rule in raw_grammar:
            [non_terminal, raw_rule_expansions] = rule.rstrip('\n').split('::=')

            rule_expansions = []
            for production_rule in raw_rule_expansions.split('|'):
                rule_expansions.append([(symbol.rstrip().lstrip().replace('<', '').replace('>', ''), \
                                        '<' in symbol) for symbol in
                                        production_rule.rstrip().lstrip().split(' ')])
            grammar[non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')] = rule_expansions

            if start_symbol is None:
                start_symbol = non_terminal.rstrip().lstrip().replace('<', '').replace('>', '')

        return grammar


    def get_grammar(self, path):
        raw_grammar = self.read_grammar(path)
        return self.parse_grammar(raw_grammar)

    def _str_(self):
        for _key_ in self.grammar:
            productions = ''
            for production in self.grammar[_key_]:
                for symbol, terminal in production:
                    if terminal:
                        productions += ' <'+symbol+'>'
                    else:
                        productions += ' '+symbol
                productions += ' | '
            print('<'+_key_+'> ::='+productions[:-3])


    def initialise(self, start_symbol):
        genotype = {}

        self.initialise_recursive((start_symbol, True), None, genotype)

        return genotype


    def initialise_recursive(self, symbol, prev_nt, genotype):
        symbol, non_terminal = symbol

        if non_terminal:
            expansion_possibility = randint(0, len(self.grammar[symbol])-1)

            if symbol not in genotype:
                genotype[symbol] = [{'ge': expansion_possibility, 'ga': {}}]
            else:
                genotype[symbol].append({'ge': expansion_possibility, 'ga': {}})

            add_reals_idx = len(genotype[symbol])-1
            for sym in self.grammar[symbol][expansion_possibility]:
                self.initialise_recursive(sym, (symbol, add_reals_idx), genotype)
        else:
            if '[' in symbol and ']' in symbol:
                genotype_key, genotype_idx = prev_nt

                [var_name, var_type, num_values, min_val, max_val] = symbol.replace('[', '')\
                                                                           .replace(']', '')\
                                                                           .split(',')

                num_values = int(num_values)
                min_val, max_val = float(min_val), float(max_val)

                if var_type == 'int':
                    values = [randint(min_val, max_val) for _ in range(num_values)]
                elif var_type == 'float':
                    values = [uniform(min_val, max_val) for _ in range(num_values)]

                genotype[genotype_key][genotype_idx]['ga'][var_name] = (var_type, min_val,
                                                                        max_val, values) 


    def decode(self, start_symbol, genotype):
        read_codons = dict.fromkeys(genotype.keys(), 0)
        phenotype = self.decode_recursive((start_symbol, True),
                                                     read_codons, genotype, '')

        return phenotype.lstrip().rstrip()


    def decode_recursive(self, symbol, read_integers, genotype, phenotype):
        symbol, non_terminal = symbol

        if non_terminal:
            if symbol not in read_integers:
                read_integers[symbol] = 0
                genotype[symbol] = []

            if len(genotype[symbol]) <= read_integers[symbol]:
                ge_expansion_integer = randint(0, len(self.grammar[symbol])-1)
                genotype[symbol].append({'ge': ge_expansion_integer, 'ga': {}})

            current_nt = read_integers[symbol]
            expansion_integer = genotype[symbol][current_nt]['ge']
            read_integers[symbol] += 1
            expansion = self.grammar[symbol][expansion_integer]

            used_terminals = []
            for sym in expansion:
                if sym[1]:
                    phenotype = self.decode_recursive(sym, read_integers, genotype, phenotype)
                else:
                    if '[' in sym[0] and ']' in sym[0]:
                        [var_name, var_type, var_num_values, var_min, var_max] = sym[0].replace('[', '')\
                                                                                       .replace(']', '')\
                                                                                       .split(',')
                        if var_name not in genotype[symbol][current_nt]['ga']:
                            var_num_values = int(var_num_values)
                            var_min, var_max = float(var_min), float(var_max)

                            if var_type == 'int':
                                values = [randint(var_min, var_max) for _ in range(var_num_values)]
                            elif var_type == 'float':
                                values = [uniform(var_min, var_max) for _ in range(var_num_values)]

                            genotype[symbol][current_nt]['ga'][var_name] = (var_type, var_min,
                                                                            var_max, values)

                        values = genotype[symbol][current_nt]['ga'][var_name][-1]

                        phenotype += ' %s:%s' % (var_name, ','.join(map(str, values)))

                        used_terminals.append(var_name)
                    else:
                        phenotype += ' '+sym[0]

            unused_terminals = list(set(list(genotype[symbol][current_nt]['ga'].keys()))\
                                    -set(used_terminals))
            if unused_terminals:
                for name in used_terminals:
                    del genotype[symbol][current_nt]['ga'][name]

        return phenotype



