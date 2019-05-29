from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder, gzip, math
import numpy as np
import codecs
from linalg import *


class MSTParserLSTM:
    def __init__(self, pos, dep_rels, sem_rels, w2i, l2i, chars, options):
        self.model = Model()
        self.PAD = 1
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'leaky': (lambda x: bmax(.1 * x, x))}
        self.activation = self.activations[options.activation]
        self.vocab = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.lemma_vocab = {lemma: ind + 2 for lemma, ind in l2i.iteritems()}
        self.pos = {word: ind + 2 for ind, word in enumerate(pos)}
        self.dep_rels = {dep_rel: ind + 1 for ind, dep_rel in enumerate(dep_rels)}
        self.sem_rels = {sem_rel: ind + 1 for ind, sem_rel in enumerate(sem_rels)}
        self.chars = {c: i + 2 for i, c in enumerate(chars)}
        self.root_id = self.dep_rels['root']
        self.idep_rels = ['PAD'] + dep_rels
        self.isem_rels = ['PAD'] + sem_rels
        self.PAD_REL = 0

        # character recurrent layer
        edim = options.char_dim
        if options.use_char:
            self.clookup = self.model.add_lookup_parameters((len(chars) + 2, options.ce))
            self.char_lstm = BiRNNBuilder(1, options.ce, edim, self.model, VanillaLSTMBuilder)

        # embedding layer
        self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim))
        self.lemlookup = self.model.add_lookup_parameters((len(l2i) + 2, edim))
        self.elookup = None
        if options.external_embedding is not None:
            external_embedding_fp = gzip.open(options.external_embedding, 'r')
            external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                  external_embedding_fp if len(line.split(' ')) > 2}
            external_embedding_fp.close()
            self.evocab = {word: i + 2 for i, word in enumerate(external_embedding)}

            edim = len(external_embedding.values()[0])
            assert edim == options.extr_dim
            self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 2, edim))
            self.elookup.set_updated(False)
            self.elookup.init_row(0, [0] * edim)
            for word in external_embedding.keys():
                self.elookup.init_row(self.evocab[word], external_embedding[word])
                if word == '_UNK_':
                    self.elookup.init_row(0, external_embedding[word])

            print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(external_embedding), \
                'words, number of training words', len(w2i) + 2
        self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))

        # recurrent layer (SHARED BiLSTM in case of having task-specific recurrent layer)
        input_dim = edim + options.pe if self.options.use_pos else edim
        self.deep_lstms = BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model, VanillaLSTMBuilder)
        for i in range(len(self.deep_lstms.builder_layers)):
            builder = self.deep_lstms.builder_layers[i]
            b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
            b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
            self.deep_lstms.builder_layers[i] = (b0, b1)

        if options.task == 'multi' and options.task_specific_recurrent_layer:
            self.sem_deep_lstms = BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model,
                                               VanillaLSTMBuilder)
            for i in range(len(self.sem_deep_lstms.builder_layers)):
                builder = self.sem_deep_lstms.builder_layers[i]
                b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
                b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
                self.sem_deep_lstms.builder_layers[i] = (b0, b1)

        # initializer for the syntax and semantic higher layers.
        fnn_dim = options.rnn * 2
        if self.options.task == 'multi' and self.options.task_specific_recurrent_layer:
            fnn_dim = options.rnn * 4

        w_mlp_arc = orthonormal_initializer(options.arc_mlp, fnn_dim)
        w_mlp_label = orthonormal_initializer(options.label_mlp, fnn_dim)

        self.need_syn_mlp = True if (self.options.task == 'syntax' or self.options.task == 'sem' or
                                     (
                                         self.options.task == 'multi' and self.options.sharing_mode != 'shared')) else False
        self.need_sem_mlp = True if (self.options.task == 'sem' or
                                     (
                                         self.options.task == 'multi' and self.options.sharing_mode != 'shared')) else False
        self.need_mtl_mlp = True if (
            self.options.task == 'multi' and self.options.sharing_mode != 'separate') else False

        if self.need_syn_mlp:
            # higher layers for syntax
            self.syn_arc_mlp_head = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                              init=NumpyInitializer(w_mlp_arc))
            self.syn_arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.syn_label_mlp_head = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                                init=NumpyInitializer(w_mlp_label))
            self.syn_label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))
            self.syn_arc_mlp_dep = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                             init=NumpyInitializer(w_mlp_arc))
            self.syn_arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.syn_label_mlp_dep = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                               init=NumpyInitializer(w_mlp_label))
            self.syn_label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))

        if self.need_sem_mlp:
            # higher layers for semantics
            self.sem_arc_mlp_head = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                              init=NumpyInitializer(w_mlp_arc))
            self.sem_arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.sem_label_mlp_head = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                                init=NumpyInitializer(w_mlp_label))
            self.sem_label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))
            self.sem_arc_mlp_dep = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                             init=NumpyInitializer(w_mlp_arc))
            self.sem_arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.sem_label_mlp_dep = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                               init=NumpyInitializer(w_mlp_label))
            self.sem_label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))

        if self.need_mtl_mlp:
            # higher layers for MTL
            self.mtl_arc_mlp_head = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                              init=NumpyInitializer(w_mlp_arc))
            self.mtl_arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.mtl_label_mlp_head = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                                init=NumpyInitializer(w_mlp_label))
            self.mtl_label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))
            self.mtl_arc_mlp_dep = self.model.add_parameters((options.arc_mlp, fnn_dim),
                                                             init=NumpyInitializer(w_mlp_arc))
            self.mtl_arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init=ConstInitializer(0))
            self.mtl_label_mlp_dep = self.model.add_parameters((options.label_mlp, fnn_dim),
                                                               init=NumpyInitializer(w_mlp_label))
            self.mtl_label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init=ConstInitializer(0))

        self.sem_w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp + 1), init=ConstInitializer(0))
        self.sem_u_label = self.model.add_parameters(
            (len(self.isem_rels) * (options.label_mlp + 1), options.label_mlp + 1),
            init=ConstInitializer(0))
        self.syn_w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp + 1), init=ConstInitializer(0))
        self.syn_u_label = self.model.add_parameters(
            (len(self.idep_rels) * (options.label_mlp + 1), options.label_mlp + 1),
            init=ConstInitializer(0))

        # dropout mask for input layers (word, external, POS, character)
        # dropout mask for word, external embeddings and Character is different from that of POS
        def _dropout_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.input_emb_dropout, batch_size).astype(np.float32)
                if self.options.use_pos:
                    tag_mask = np.random.binomial(1, 1. - self.options.input_emb_dropout, batch_size).astype(np.float32)
                    scale = 3. / (2. * word_mask + tag_mask + 1e-12) if not self.options.use_char else 5. / (
                        4. * word_mask + tag_mask + 1e-12)
                    word_mask *= scale
                    tag_mask *= scale
                    word_mask = inputTensor(word_mask, batched=True)
                    tag_mask = inputTensor(tag_mask, batched=True)
                    ret.append((word_mask, tag_mask))
                else:
                    scale = 2. / (2. * word_mask + 1e-12) if not self.options.use_char else 4. / (
                        4. * word_mask + 1e-12)
                    word_mask *= scale
                    word_mask = inputTensor(word_mask, batched=True)
                    ret.append(word_mask)
            return ret

        self.generate_emb_dropout_mask = _dropout_mask_generator

    def bilinear(self, M, W, H, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
        if bias_x:
            M = concatenate([M, inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        if bias_y:
            H = concatenate([H, inputTensor(np.ones((1, seq_len), dtype=np.float32))])

        nx, ny = input_size + bias_x, input_size + bias_y
        lin = W * M
        if num_outputs > 1:
            lin = reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
        blin = transpose(H) * lin
        if num_outputs > 1:
            blin = reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
        return blin

    def __evaluate(self, H, M):
        M2 = concatenate([M, inputTensor(np.ones((1, M.dim()[0][1]), dtype=np.float32))])
        return transpose(H) * (self.syn_w_arc.expr() * M2)

    def __evaluateLabel(self, i, j, HL, ML):
        H2 = concatenate([HL, inputTensor(np.ones((1, HL.dim()[0][1]), dtype=np.float32))])
        M2 = concatenate([ML, inputTensor(np.ones((1, ML.dim()[0][1]), dtype=np.float32))])
        h, m = transpose(H2), transpose(M2)
        return reshape(transpose(h[i]) * self.syn_u_label.expr(), (len(self.idep_rels), self.options.label_mlp + 1)) * \
               m[j]

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def bi_rnn(self, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
        # shared BiLSTM
        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fb.set_dropouts(dropout_x, dropout_h)
            bb.set_dropouts(dropout_x, dropout_h)
            if batch_size is not None:
                fb.set_dropout_masks(batch_size)
                bb.set_dropout_masks(batch_size)
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def sem_bi_rnn(self, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
        # semantic BiLSTM
        for fb, bb in self.sem_deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fb.set_dropouts(dropout_x, dropout_h)
            bb.set_dropouts(dropout_x, dropout_h)
            if batch_size is not None:
                fb.set_dropout_masks(batch_size)
                bb.set_dropout_masks(batch_size)
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def fnn_syn(self, h, train):
        H = self.activation(affine_transform([self.syn_arc_mlp_head_b.expr(), self.syn_arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.syn_arc_mlp_dep_b.expr(), self.syn_arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.syn_label_mlp_head_b.expr(), self.syn_label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.syn_label_mlp_dep_b.expr(), self.syn_label_mlp_dep.expr(), h]))

        arc_dropout = self.options.arc_dropout
        label_dropout = self.options.label_dropout

        if train:
            H, M, HL, ML = dropout_dim(H, 1, arc_dropout), dropout_dim(M, 1, arc_dropout), dropout_dim(HL, 1,
                                                                                                       label_dropout), dropout_dim(
                ML, 1, label_dropout)
        return H, M, HL, ML

    def fnn_sem(self, h, train):
        H = self.activation(affine_transform([self.sem_arc_mlp_head_b.expr(), self.sem_arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.sem_arc_mlp_dep_b.expr(), self.sem_arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.sem_label_mlp_head_b.expr(), self.sem_label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.sem_label_mlp_dep_b.expr(), self.sem_label_mlp_dep.expr(), h]))

        arc_dropout = self.options.arc_dropout
        label_dropout = self.options.label_dropout

        if train:
            H, M, HL, ML = dropout_dim(H, 1, arc_dropout), \
                           dropout_dim(M, 1, arc_dropout), \
                           dropout_dim(HL, 1, label_dropout), \
                           dropout_dim(ML, 1, label_dropout)
        return H, M, HL, ML

    def fnn_mtl(self, h, train):
        H = self.activation(affine_transform([self.mtl_arc_mlp_head_b.expr(), self.mtl_arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.mtl_arc_mlp_dep_b.expr(), self.mtl_arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.mtl_label_mlp_head_b.expr(), self.mtl_label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.mtl_label_mlp_dep_b.expr(), self.mtl_label_mlp_dep.expr(), h]))

        arc_dropout = self.options.arc_dropout
        label_dropout = self.options.label_dropout

        if train:
            H, M, HL, ML = dropout_dim(H, 1, arc_dropout), dropout_dim(M, 1, arc_dropout), dropout_dim(HL, 1,
                                                                                                       label_dropout), dropout_dim(
                ML, 1, label_dropout)
        return H, M, HL, ML

    def fnn_mtl_sum(self, h, task, train):
        if task == 'syntax':
            H = self.activation(affine_transform([self.mtl_arc_mlp_head_b.expr() + self.syn_arc_mlp_head_b.expr(),
                                                  self.mtl_arc_mlp_head.expr() + self.syn_arc_mlp_head.expr(), h]))
            M = self.activation(affine_transform([self.mtl_arc_mlp_dep_b.expr() + self.syn_arc_mlp_dep_b.expr(),
                                                  self.mtl_arc_mlp_dep.expr() + self.syn_arc_mlp_dep.expr(), h]))
            HL = self.activation(affine_transform([self.mtl_label_mlp_head_b.expr() + self.syn_label_mlp_head_b.expr(),
                                                   self.mtl_label_mlp_head.expr() + self.syn_label_mlp_head.expr(), h]))
            ML = self.activation(affine_transform([self.mtl_label_mlp_dep_b.expr() + self.syn_label_mlp_dep_b.expr(),
                                                   self.mtl_label_mlp_dep.expr() + self.syn_label_mlp_dep.expr(), h]))
        elif task == 'sem':
            H = self.activation(affine_transform([self.mtl_arc_mlp_head_b.expr() + self.sem_arc_mlp_head_b.expr(),
                                                  self.mtl_arc_mlp_head.expr() + self.sem_arc_mlp_head.expr(), h]))
            M = self.activation(affine_transform([self.mtl_arc_mlp_dep_b.expr() + self.sem_arc_mlp_dep_b.expr(),
                                                  self.mtl_arc_mlp_dep.expr() + self.sem_arc_mlp_dep.expr(), h]))
            HL = self.activation(affine_transform([self.mtl_label_mlp_head_b.expr() + self.sem_label_mlp_head_b.expr(),
                                                   self.mtl_label_mlp_head.expr() + self.sem_label_mlp_head.expr(), h]))
            ML = self.activation(affine_transform([self.mtl_label_mlp_dep_b.expr() + self.sem_label_mlp_dep_b.expr(),
                                                   self.mtl_label_mlp_dep.expr() + self.sem_label_mlp_dep.expr(), h]))

        arc_dropout = self.options.arc_dropout
        label_dropout = self.options.label_dropout

        if train:
            H, M, HL, ML = dropout_dim(H, 1, arc_dropout), dropout_dim(M, 1, arc_dropout), dropout_dim(HL, 1,
                                                                                                       label_dropout), dropout_dim(
                ML, 1, label_dropout)
        return H, M, HL, ML

    def char_lstm_output(self, cembed, train=False, batch_size=None):
        fb, bb = self.char_lstm.builder_layers[0][0], self.char_lstm.builder_layers[0][1]
        f, b = fb.initial_state(), bb.initial_state()
        char_fwd, char_bckd = f.transduce(cembed)[-1], b.transduce(reversed(cembed))[-1]

        return (char_fwd, char_bckd)

    def recurrent_layer(self, sens, train):
        words, lemmas, pwords, pos, chars = sens[0], sens[1], sens[2], sens[3], sens[8]
        if self.options.use_char:
            cembed = [lookup_batch(self.clookup, c) for c in chars]
            char_fwd, char_bckd = self.char_lstm_output(cembed, train, words.shape[0])
            crnn = reshape(concatenate_cols([char_fwd, char_bckd]), (self.options.we, words.shape[0] * words.shape[1]))
            cnn_reps = [list() for _ in range(len(words))]
            for i in range(words.shape[0]):
                cnn_reps[i] = pick_batch(crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

            if self.options.use_lemma:
                wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.lemlookup, lemmas[i]) + lookup_batch(
                    self.elookup, pwords[i]) + cnn_reps[i] for i
                          in range(len(words))]
            else:
                wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.elookup, pwords[i]) + cnn_reps[i] for
                          i
                          in range(len(words))]
        else:
            if self.options.use_lemma:
                wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.lemlookup, lemmas[i]) + lookup_batch(
                    self.elookup, pwords[i]) for i in
                          range(len(words))]
            else:
                wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.elookup, pwords[i]) for i in
                          range(len(words))]

        posembed = [lookup_batch(self.plookup, pos[i]) for i in range(len(pos))] if self.options.use_pos else None

        if not train:
            inputs = [concatenate([w, pos]) for w, pos in zip(wembed, posembed)] if self.options.use_pos else wembed
        else:
            emb_dropout_masks = self.generate_emb_dropout_mask(words.shape[0], words.shape[1])
            inputs = [concatenate([cmult(w, wm), cmult(pos, posm)]) for w, pos, (wm, posm) in
                      zip(wembed, posembed, emb_dropout_masks)] if self.options.use_pos \
                else [cmult(w, wm) for w, wm in zip(wembed, emb_dropout_masks)]

        bilstm_recur_dropout = self.options.bilstm_recur_dropout
        bilstm_ff_dropout = self.options.bilstm_ff_dropout
        h_out = self.bi_rnn(inputs, words.shape[1], bilstm_ff_dropout if train else 0,
                            bilstm_recur_dropout if train else 0)
        h = dropout_dim(concatenate_cols(h_out), 1, bilstm_recur_dropout) if train else \
            concatenate_cols(h_out)

        sem_h = None
        if self.options.task_specific_recurrent_layer:
            sem_h_out = self.sem_bi_rnn(inputs, words.shape[1], bilstm_ff_dropout if train else 0,
                                        bilstm_recur_dropout if train else 0)
            sem_h = dropout_dim(concatenate_cols(sem_h_out), 1, bilstm_recur_dropout) if train else \
                concatenate_cols(sem_h_out)

        return h, sem_h

    def build_syntax_graph(self, mini_batch, t=1):
        h = self.recurrent_layer(mini_batch, train=True)
        arc_scores, rel_scores = self.get_syntax_scores(h, mini_batch, train=True)
        flat_scores = reshape(arc_scores, (mini_batch[0].shape[0],), mini_batch[0].shape[0] * mini_batch[0].shape[1])
        flat_rel_scores = reshape(rel_scores, (mini_batch[0].shape[0], len(self.idep_rels)),
                                  mini_batch[0].shape[0] * mini_batch[0].shape[1])

        masks = np.reshape(mini_batch[-1], (-1,), 'F')
        mask_1D_tensor = inputTensor(masks, batched=True)
        n_tokens = np.sum(masks)
        heads = np.reshape(mini_batch[4], (-1,), 'F')
        partial_rel_scores = pick_batch(flat_rel_scores, heads)
        gold_relations = np.reshape(mini_batch[5], (-1,), 'F')
        arc_losses = pickneglogsoftmax_batch(flat_scores, heads)
        arc_loss = sum_batches(arc_losses * mask_1D_tensor) / n_tokens
        rel_losses = pickneglogsoftmax_batch(partial_rel_scores, gold_relations)
        rel_loss = sum_batches(rel_losses * mask_1D_tensor) / n_tokens
        err = 0.5 * (arc_loss + rel_loss)
        err.scalar_value()
        loss = err.value()
        err.backward()
        self.trainer.update()
        renew_cg()
        return t + 1, loss

    def get_syntax_scores(self, h, mini_batch, train):
        H, M, HL, ML = self.fnn_syn(h, train)
        arc_scores = self.bilinear(M, self.syn_w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0],
                                   mini_batch[0].shape[1], 1, True, False)
        rel_scores = self.bilinear(ML, self.syn_u_label.expr(), HL, self.options.label_mlp, mini_batch[0].shape[0],
                                   mini_batch[0].shape[1], len(self.idep_rels), True, True)
        return arc_scores, rel_scores

    def build_semantic_graph(self, mini_batch, t=1):
        h, _ = self.recurrent_layer(mini_batch, train=True)
        head_scores, rel_scores, _, _ = self.get_sem_scores(h, mini_batch, train=True)
        head_loss, rel_loss = self.sem_loss(mini_batch, head_scores, rel_scores)
        coef = self.options.interpolation_coef
        err = coef * rel_loss + (1 - coef) * head_loss
        err.scalar_value()
        loss = err.value()
        if math.isnan(loss):
            print 'loss value:', loss
            print 'head_loss, rel_loss', head_loss.value(), rel_loss.value()
            print 'n_head_tokens, n_rel_tokens', n_head_tokens, n_rel_tokens
        err.backward()
        self.trainer.update()
        renew_cg()
        return t + 1, loss

    def get_sem_scores(self, h, mini_batch, train):
        H, M, HL, ML = self.fnn_sem(h, train)
        head_scores = self.bilinear(M, self.sem_w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0],
                                    mini_batch[0].shape[1], 1, True, False)
        rel_scores = self.sem_bilinear(HL, ML, mini_batch[0].shape[0], mini_batch[-2])
        return head_scores, rel_scores, HL, ML

    def sem_bilinear(self, HL, ML, seq_len, rel_h_m_indices):
        """
        Similar to bilinear, but for sparse computation (memory efficient) of semantic roles.
        """
        head_indices, modifier_indices = rel_h_m_indices
        if len(head_indices) == 0:
            return None
        HL_w_b = concatenate([HL, inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        ML_w_b = concatenate([ML, inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        HL_reshape = transpose(reshape(HL_w_b, (HL_w_b.dim()[0][0], HL_w_b.dim()[0][1] * HL_w_b.dim()[1])))
        ML_reshape = transpose(reshape(ML_w_b, (ML_w_b.dim()[0][0], ML_w_b.dim()[0][1] * ML_w_b.dim()[1])))
        HL_to_use = concatenate_to_batch([HL_reshape[hi] for hi in head_indices])
        ML_to_use = concatenate_to_batch([ML_reshape[mi] for mi in modifier_indices])
        num_outputs = len(self.isem_rels)

        lin = self.sem_u_label.expr() * ML_to_use
        lin = reshape(lin, (HL_w_b.dim()[0][0], num_outputs), batch_size=len(modifier_indices))
        blin = transpose(HL_to_use) * lin
        bilin_scores = reshape(blin, (num_outputs,), batch_size=len(modifier_indices))
        return bilin_scores

    def get_scores(self, H, M, HL, ML, mini_batch, mtl_task):
        if mtl_task == 'sem':
            head_scores = self.bilinear(M, self.sem_w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0],
                                        mini_batch[0].shape[1], 1, True, False)
            rel_scores = self.sem_bilinear(HL, ML, mini_batch[0].shape[0], mini_batch[-2])
        elif mtl_task == 'syntax':
            head_scores = self.bilinear(M, self.syn_w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0],
                                        mini_batch[0].shape[1], 1, True, False)
            rel_scores = self.bilinear(ML, self.syn_u_label.expr(), HL, self.options.label_mlp, mini_batch[0].shape[0],
                                       mini_batch[0].shape[1], len(self.idep_rels), True, True)

        return head_scores, rel_scores

    def get_mtl_scores(self, h, mini_batch, mlp_mode, train):
        sem_hs, sem_rs, syn_hs, syn_rs = None, None, None, None
        if mlp_mode == 'shared':
            H, M, HL, ML = self.fnn_mtl(h, train)
            sem_hs, sem_rs = self.get_scores(H, M, HL, ML, mini_batch, 'sem')
            syn_hs, syn_rs = self.get_scores(H, M, HL, ML, mini_batch, 'syntax')
        elif mlp_mode == 'separate':
            sem_hs, sem_rs, HL, ML = self.get_sem_scores(h, mini_batch, train)
            syn_hs_, syn_rs_ = self.get_syntax_scores(h, mini_batch, train)
            syn_hs = reshape(syn_hs_, (mini_batch[0].shape[0],),
                             mini_batch[0].shape[0] * mini_batch[0].shape[1])
            syn_rs = reshape(syn_rs_, (mini_batch[0].shape[0], len(self.idep_rels)),
                             mini_batch[0].shape[0] * mini_batch[0].shape[1])
        elif mlp_mode == 'sum':
            # shared scores
            H_syn, M_syn, HL_syn, ML_syn = self.fnn_mtl_sum(h, 'syntax', train)
            H_sem, M_sem, HL, ML = self.fnn_mtl_sum(h, 'sem', train)

            sem_hs, sem_rs = self.get_scores(H_sem, M_sem, HL_sem, ML_sem, mini_batch, 'sem')
            syn_hs, syn_rs = self.get_scores(H_syn, M_syn, HL_syn, ML_syn, mini_batch, 'syntax')

        return sem_hs, sem_rs, syn_hs, syn_rs, HL, ML

    def sem_loss(self, mini_batch, sem_head_scores, sem_rel_scores):
        heads = np.reshape(mini_batch[6], (-1,), 'F')
        heads_tensor = inputTensor(heads, batched=True)
        head_masks = np.reshape(mini_batch[-3], (-1,), 'F')
        indices_to_use_for_head = [i[0] for (i, mask) in np.ndenumerate(head_masks) if mask == 1]
        if len(indices_to_use_for_head) == 0:
            return
        n_head_tokens = len(indices_to_use_for_head)

        flat_head_scores = reshape(sem_head_scores, (1,), heads.shape[0])
        flat_head_scores_to_use = pick_batch(reshape(flat_head_scores, (heads.shape[0],)), indices_to_use_for_head)
        heads_tensor_to_use = pick_batch(reshape(heads_tensor, (heads.shape[0],)), indices_to_use_for_head)
        flat_head_probs = logistic(flat_head_scores_to_use) + 1e-12
        head_losses = binary_log_loss(flat_head_probs, heads_tensor_to_use)
        head_loss = sum_batches(head_losses) / n_head_tokens

        gold_rels = mini_batch[7]
        if len(gold_rels) > 0:
            rel_losses = pickneglogsoftmax_batch(sem_rel_scores, gold_rels)
            rel_loss = sum_batches(rel_losses) / len(gold_rels)
        else:
            rel_loss = 0

        return head_loss, rel_loss

    def syn_loss(self, mini_batch, syn_hs, syn_rs):
        masks = np.reshape(mini_batch[-1], (-1,), 'F')
        n_tokens = np.sum(masks)
        if n_tokens == 0:
            return
        flat_scores = reshape(syn_hs, (mini_batch[0].shape[0],), mini_batch[0].shape[0] * mini_batch[0].shape[1])
        flat_rel_scores = reshape(syn_rs, (mini_batch[0].shape[0], len(self.idep_rels)),
                                  mini_batch[0].shape[0] * mini_batch[0].shape[1])
        mask_1D_tensor = inputTensor(masks, batched=True)
        heads = np.reshape(mini_batch[4], (-1,), 'F')
        partial_rel_scores = pick_batch(flat_rel_scores, heads)
        gold_relations = np.reshape(mini_batch[5], (-1,), 'F')
        arc_losses = pickneglogsoftmax_batch(flat_scores, heads)
        arc_loss = sum_batches(arc_losses * mask_1D_tensor) / n_tokens
        rel_losses = pickneglogsoftmax_batch(partial_rel_scores, gold_relations)
        rel_loss = sum_batches(rel_losses * mask_1D_tensor) / n_tokens
        return arc_loss, rel_loss

    def build_mtl_graph(self, mini_batch, sharing_mode, t=1):
        shared_h, sem_h = self.recurrent_layer(mini_batch, train=True)
        if self.options.task_specific_recurrent_layer:
            h = concatenate([shared_h, sem_h])
        else:
            h = shared_h

        sem_hs, sem_rs, syn_hs, syn_rs, _, _ = self.get_mtl_scores(h, mini_batch, sharing_mode, train=True)
        sem_loss = self.sem_loss(mini_batch, sem_hs, sem_rs)
        sem_head_loss, sem_rel_loss = (sem_loss[0], sem_loss[1]) if sem_loss is not None else (None, None)
        syn_loss = self.syn_loss(mini_batch, syn_hs, syn_rs)
        syn_head_loss, syn_rel_loss = (syn_loss[0], syn_loss[1]) if syn_loss is not None else (None, None)

        coef = self.options.interpolation_coef
        mtl_coef = self.options.mtl_coef
        sem_err = coef * sem_rel_loss + (1 - coef) * sem_head_loss if sem_loss else 0
        # todo change to have better performance
        if self.options.syn_head_loss_bp:
            syn_err = (syn_rel_loss + syn_head_loss) / 2 if syn_loss else 0
        else:
            syn_err = syn_rel_loss if syn_loss else 0
        err = mtl_coef * sem_err + (1 - mtl_coef) * syn_err
        err.scalar_value()
        loss = err.value()
        if math.isnan(loss):
            print 'loss value:', loss
        err.backward()
        self.trainer.update()
        renew_cg()
        return t + 1, loss

    def decode(self, mini_batch):
        shared_h, sem_h = self.recurrent_layer(mini_batch, train=False)
        if self.options.task == 'multi' and self.options.task_specific_recurrent_layer:
            h = concatenate([shared_h, sem_h])
        else:
            h = shared_h

        if self.options.task == 'multi' and self.options.sharing_mode == "shared":
            sem_head_scores, _, syn_head_scores, syn_rel_scores, HL, ML = self.get_mtl_scores(h, mini_batch,
                                                                                              self.options.sharing_mode,
                                                                                              train=False)
            flat_syn_head_scores = reshape(syn_head_scores, (mini_batch[0].shape[0],),
                                           mini_batch[0].shape[0] * mini_batch[0].shape[1])
            flat_syn_rel_scores = reshape(syn_rel_scores, (mini_batch[0].shape[0], len(self.idep_rels)),
                                          mini_batch[0].shape[0] * mini_batch[0].shape[1])
        else:
            syn_head_scores, syn_rel_scores = self.get_syntax_scores(h, mini_batch, train=False)
            flat_syn_head_scores = reshape(syn_head_scores, (mini_batch[0].shape[0],),
                                           mini_batch[0].shape[0] * mini_batch[0].shape[1])
            flat_syn_rel_scores = reshape(syn_rel_scores, (mini_batch[0].shape[0], len(self.idep_rels)),
                                          mini_batch[0].shape[0] * mini_batch[0].shape[1])
            sem_head_scores, _, HL, ML = self.get_sem_scores(h, mini_batch, train=False)

        # Syntax
        syntax_arc_probs = np.transpose(np.reshape(softmax(flat_syn_head_scores).npvalue(), (
            mini_batch[0].shape[0], mini_batch[0].shape[0], mini_batch[0].shape[1]), 'F'))
        syntax_rel_probs = np.transpose(np.reshape(softmax(transpose(flat_syn_rel_scores)).npvalue(),
                                                   (len(self.idep_rels), mini_batch[0].shape[0], mini_batch[0].shape[0],
                                                    mini_batch[0].shape[1]), 'F'))
        syntax_outputs = []

        for msk, syntax_arc_prob, syntax_rel_prob in zip(np.transpose(mini_batch[-1]), syntax_arc_probs,
                                                         syntax_rel_probs):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = decoder.arc_argmax(syntax_arc_prob, sent_len, msk)
            syntax_rel_prob = syntax_rel_prob[np.arange(len(arc_pred)), arc_pred]
            syntax_rel_pred = decoder.rel_argmax(syntax_rel_prob, sent_len, self.PAD_REL, self.root_id)
            syntax_outputs.append((arc_pred[1:sent_len], syntax_rel_pred[1:sent_len]))

        # Semantics
        sem_head_score_values = sem_head_scores.npvalue()
        if sem_head_scores.dim()[1] == 1:
            sem_head_score_values = np.reshape(sem_head_score_values,
                                               (sem_head_score_values.shape[0], sem_head_score_values.shape[1], 1))

        assert sem_head_score_values.shape == mini_batch[6].shape
        sem_mask = mini_batch[-3]
        sem_heads = np.array(
            [np.array(
                [
                    np.array(
                        [
                            1 if (sem_head_score_values[i][j][k] >= 0 and sem_mask[i][j][k] == 1) else 0 for k in
                            range(sem_head_score_values.shape[2])
                        ]
                    )
                    for j in range(sem_head_score_values.shape[1])
                ]
            )
                for i in range(sem_head_score_values.shape[0])
            ]
        )

        modifier_indices, head_indices = [], []
        for i in range(sem_head_score_values.shape[0]):
            for j in range(sem_head_score_values.shape[1]):
                for k in range(sem_head_score_values.shape[2]):
                    if sem_heads[i][j][k] == 1:
                        modifier_idx = k * sem_head_score_values.shape[0] + i
                        head_idx = k * sem_head_score_values.shape[0] + j
                        modifier_indices.append(modifier_idx)
                        head_indices.append(head_idx)
        sem_rel_scores = self.sem_bilinear(HL, ML, sem_head_score_values.shape[0], [head_indices, modifier_indices])
        sem_rel_argmax = np.zeros(sem_heads.shape, dtype=np.int32)
        if sem_rel_scores is not None:
            sem_rels = np.argmax(sem_rel_scores.npvalue(), axis=0)
            print 'sem_rels shape: '+ str(sem_rels.shape)
            print 'sem_rel_argmax shape: '+ str(sem_rel_argmax.shape)
            print 'sem_rel_scores shape: '+ str(sem_rel_scores.dime())
            c = 0
            for i in range(sem_head_score_values.shape[0]):
                for j in range(sem_head_score_values.shape[1]):
                    for k in range(sem_head_score_values.shape[2]):
                        if sem_heads[i][j][k] == 1:
                            sem_rel_argmax[i][j][k] = sem_rels[c]
                            c += 1
        renew_cg()
        return syntax_outputs, sem_heads, sem_rel_argmax
