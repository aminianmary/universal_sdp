from collections import Counter, defaultdict
import re, codecs, sys, random
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')


class ConllEntry:
    def __init__(self, id, form, lemma, pos, fpos, feats=None, head=None, dep_relation=None, sem_deps=dict(),
                 misc=None):
        self.id = id
        self.form = form.lower()  # assuming everything is lowercased.
        self.norm = normalize(form)
        self.fpos = fpos.upper()
        self.pos = pos.upper()
        self.head = head
        self.dep_relation = dep_relation

        self.lemma = lemma
        self.feats = feats
        self.sem_deps = sem_deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.pred_deps = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.fpos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation]
        sd_value = '_'
        if len(self.sem_deps) > 0:
            for sd in self.sem_deps.keys():
                if sd == -1:
                    sd_value = '?'
                else:
                    sd_value = '|'.join(str(self.sem_deps[sd]) + ':' + self.sem_deps[sd])
        values.append(sd_value)
        values.append(self.misc)
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path, min_count):
    wordsCount = Counter()
    lemmaCount = Counter()
    posCount = Counter()
    depRelCount = Counter()
    semRelCount = Counter()
    chars = set()
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            lemmaCount.update([node.lemma for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            depRelCount.update([node.dep_relation for node in sentence if isinstance(node, ConllEntry)])
            for node in sentence:
                if isinstance(node, ConllEntry):
                    sd_dic = node.sem_deps
                    for sd in sd_dic.keys():
                        semRelCount.update([sd_dic[sd]])
            for node in sentence:
                if isinstance(node, ConllEntry):
                    for c in list(node.norm):
                        chars.add(c.lower())

    words = set()
    for w in wordsCount.keys():
        if wordsCount[w] >= min_count:
            words.add(w)
    lemmas = set()
    for w in lemmaCount.keys():
        if lemmaCount[w] >= min_count:
            lemmas.add(w)
    return (
        {w: i for i, w in enumerate(words)}, {l: i for i, l in enumerate(lemmas)} ,list(posCount.keys()), list(depRelCount.keys()), list(semRelCount.keys()),
        list(chars))


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-FPOS', '_', 0, 'root', dict(), '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '' or line.strip().startswith("#"):
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                sd = tok[8].strip()
                sd_dic = get_sem_deps(sd)
                if tok[6] == '_' or tok[6] == '?':
                    dep_head =  -1
                else:
                    dep_head = int(tok[6])
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5],
                                         dep_head, tok[7], sd_dic, tok[9]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with codecs.open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + u'\n')
            fh.write('\n')


def eval(gold, predicted):
    # evaluation done in the CoNLLU data format
    correct_syn_dep, correct_syn_l, all_syn_deps = 0, 0, 0
    all_gold_sem_deps, all_sys_sem_deps = 0, 0
    correct_sem_dep, correct_sem_l = 0, 0
    prediction_reader = open(predicted, 'r')

    for gold_line in open(gold, 'r'):
        prediction_line = prediction_reader.readline().strip()
        gold_line = gold_line.strip()
        if gold_line != '' and gold_line[0] != '#':
            prediction_fields = prediction_line.split('\t')
            gold_fields = gold_line.split('\t')
            gold_syn_head = gold_fields[6]
            gold_syn_rel = gold_fields[7]
            gold_sem_deps = get_sem_deps(gold_fields[8])
            sys_syn_head = prediction_fields[6]
            sys_syn_rel = prediction_fields[7]
            sys_sem_deps = get_sem_deps(prediction_fields[8])
            if not is_punc(prediction_fields[3]):
                # syntax
                all_syn_deps += 1
                if gold_syn_head == sys_syn_head:
                    correct_syn_dep += 1
                    if gold_syn_rel == sys_syn_rel:
                        correct_syn_l += 1
                # semantics
                all_gold_sem_deps += len(gold_sem_deps.keys())
                all_sys_sem_deps += len(sys_sem_deps.keys())
                h, l = eval_sdp(gold_sem_deps, sys_sem_deps)
                correct_sem_dep += h
                correct_sem_l += l

    uas, las = 100 * float(correct_syn_dep) / all_syn_deps, 100 * float(correct_syn_l) / all_syn_deps
    lf, uf = 0, 0
    if all_sys_sem_deps != 0:
        lp, lr = 100 * float(correct_sem_l) / all_sys_sem_deps, 100 * float(correct_sem_l) / all_gold_sem_deps
        up, ur = 100 * float(correct_sem_dep) / all_sys_sem_deps, 100 * float(
            correct_sem_dep) / all_gold_sem_deps
        if correct_sem_l != 0:
            lf = 2 * (lp * lr) / (lp + lr)
        if correct_sem_dep != 0:
            uf = 2 * (up * ur) / (up + ur)
    return uas, las, uf, lf


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")


def get_sem_deps(sd):
    sd_dic = {}
    if sd == '?':
        sd_dic[-1] = '?'
    elif sd != '_':
        for e in sd.split('|'):
            spl = e.split(':')
            if len(spl) == 2:
                sd_dic[int(spl[0])] = spl[1]
            else:
                print spl
    return sd_dic


def eval_sdp(gold_sem_deps, sys_sem_deps):
    correct_sem_heads, correct_sem_rels = 0, 0
    for head in sys_sem_deps:
        if head in gold_sem_deps:
            correct_sem_heads += 1
            if gold_sem_deps[head] == sys_sem_deps[head]:
                correct_sem_rels += 1
    return correct_sem_heads, correct_sem_rels


def normalize(word):
    return '<num>' if numberRegex.match(word) else ('<url>' if urlRegex.match(word) else word.lower())


def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, cur_len, cur_c_len = [], 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d) <= 100) or not is_train:
                batch.append(d)
                sen_id, conll_entries = d
                cur_c_len = max(cur_c_len, max([len(w.norm) for w in conll_entries]))
                cur_len = max(cur_len, len(conll_entries))

            if cur_len * len(batch) >= model.options.batch:
                add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train)
                batch, cur_len, cur_c_len = [], 0, 0

    if len(batch) > 0:
        add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train)
        batch, cur_len = [], 0
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train):
    words = np.array([np.array(
        [model.vocab.get(batch[i][1][j].norm, 0) if j < len(batch[i][1]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    lemmas = np.array([np.array(
        [model.lemma_vocab.get(batch[i][1][j].lemma, 0) if j < len(batch[i][1]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        [model.evocab.get(batch[i][1][j].norm, 0) if j < len(batch[i][1]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pos = np.array([np.array(
        [model.pos.get(batch[i][1][j].pos, 0) if j < len(batch[i][1]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    dep_heads = np.array(
        [np.array(
            [batch[i][1][j].head if 0 < j < len(batch[i][1]) and batch[i][1][j].head >= 0 else 0 for i in range(len(batch))]) for
            j in range(cur_len)])
    dep_relations = np.array([np.array(
        [model.dep_rels.get(batch[i][1][j].dep_relation, 0) if j < len(batch[i][1]) else model.PAD_REL for i in
         range(len(batch))]) for j in range(cur_len)])

    sem_heads = np.array(
        [np.array(
            [
                np.array(
                    [
                        1 if 0 < d < len(batch[b][1]) and h in batch[b][1][d].sem_deps else 0 for b in range(len(batch))
                    ]
                )
                for h in range(cur_len)
            ]
        )
            for d in range(cur_len)
        ]
    )

    sem_head_masks = np.array(
        [np.array(
            [
                np.array(
                    [
                        1 if (0 < j < len(batch[i][1]) and (-1 not in batch[i][1][j].sem_deps or not is_train)) else 0 for i
                        in range(len(batch))
                    ]
                )
            ] * cur_len
        )
            for j in range(cur_len)
        ]
    )

    sem_rels, sem_rel_head_idx_to_use,sem_rel_mod_idx_to_use = [],[],[]
    if is_train:
        for dep in range(cur_len):
            for head in range(cur_len):
                for b in range(len(batch)):
                    if 0 < dep < len(batch[b][1]) and head in batch[b][1][dep].sem_deps:
                        sem_rels.append(model.sem_rels.get(batch[b][1][dep].sem_deps[head], 0))
                        head_index = b * cur_len + head
                        mod_index = b * cur_len + dep
                        sem_rel_head_idx_to_use.append(head_index)
                        sem_rel_mod_idx_to_use.append(mod_index)

    chars = [list() for _ in range(cur_c_len)]
    for c_pos in range(cur_c_len):
        ch = [model.PAD] * (len(batch) * cur_len)
        offset = 0
        for w_pos in range(cur_len):
            for sen_position in range(len(batch)):
                if w_pos < len(batch[sen_position][1]) and c_pos < len(batch[sen_position][1][w_pos].norm):
                    ch[offset] = model.chars.get(batch[sen_position][1][w_pos].norm[c_pos], 0)
                offset += 1
        chars[c_pos] = np.array(ch)
    chars = np.array(chars)
    dep_masks = np.array([np.array(
        [1 if 0 < j < len(batch[i][1]) and (batch[i][1][j].head >= 0 or not is_train) else 0 for i in range(len(batch))])
        for j in range(cur_len)])

    sen_ids = np.array([batch[i][0] for i in range(len(batch))])

    mini_batches.append((words, lemmas , pwords, pos, dep_heads, dep_relations, sem_heads, sem_rels, chars, sem_head_masks,
                         [sem_rel_head_idx_to_use, sem_rel_mod_idx_to_use], sen_ids, dep_masks))

def is_punc(pos):
    return pos == '.' or pos == 'PUNC' or pos == 'PUNCT' or \
           pos == "#" or pos == "''" or pos == "(" or \
           pos == "[" or pos == "]" or pos == "{" or pos == "}" or \
           pos == "\"" or pos == "," or pos == "." or pos == ":" or \
           pos == "``" or pos == "-LRB-" or pos == "-RRB-" or pos == "-LSB-" or \
           pos == "-RSB-" or pos == "-LCB-" or pos == "-RCB-" or pos == '"' or pos == ')'

def bert_features(bert_fp):
    bert_features = {}
    with open(bert_fp,'r') as reader:
        s_id = 0
        t_id = 0
        sen_bert = {}
        for line in reader:
            if line.strip() == '':
                bert_features[s_id] = sen_bert
                sen_bert = {}
                t_id = 0
                s_id +=1
            else:
                t_id +=1
                sen_bert[t_id]= [float(f) for f in line.strip().split(' ')[1].split('\t')]

    return bert_features
