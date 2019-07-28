from optparse import OptionParser
import pickle, utils, mstlstm, sys, os.path, time

def parse(parser, buckets, test_file, output_file, is_dev):
    syntax_results = list()
    sem_head_results = list()
    sem_rel_results = list()
    for mini_batch in utils.get_batches(buckets, parser, False):
        syntax_outputs, sem_heads, sem_rel_argmax = parser.decode(mini_batch, is_dev)
        for output in syntax_outputs:
            syntax_results.append(output)
        sem_head_results.append(sem_heads)
        sem_rel_results.append(sem_rel_argmax)

    arcs = reduce(lambda x, y: x + y, [list(result[0]) for result in syntax_results])
    dep_rels = reduce(lambda x, y: x + y, [list(result[1]) for result in syntax_results])
    idx = 0
    cur_sem_head_results = sem_head_results.pop(0)
    cur_sem_rel_results = sem_rel_results.pop(0)
    cur_sem_batch_num = 0
    with open(test_file) as f:
        with open(output_file, 'w') as fo:
            for line in f.readlines():
                info = line.strip().split('\t')
                if line.startswith('#'):
                    fo.write(line.strip() + '\n')
                elif info and line.strip() != '':
                    assert len(info) == 10, 'Illegal line: %s' % line
                    dep_index = int(info[0])
                    info[6] = str(arcs[idx])
                    info[7] = parser.idep_rels[dep_rels[idx]]
                    sem_info = []
                    for head_index in range(cur_sem_head_results.shape[0]):
                        if cur_sem_head_results[dep_index, head_index, cur_sem_batch_num] == 1:
                            sem_info.append(str(head_index) + ":" + parser.isem_rels[
                                cur_sem_rel_results[dep_index, head_index, cur_sem_batch_num]])
                    info[8] = '|'.join(sem_info) if len(sem_info) > 0 else '_'
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    cur_sem_batch_num += 1
                    if cur_sem_batch_num == cur_sem_head_results.shape[2]:
                        if len(sem_head_results) > 0:
                            cur_sem_head_results = sem_head_results.pop(0)
                            cur_sem_rel_results = sem_rel_results.pop(0)
                            cur_sem_batch_num = 0
                    fo.write('\n')
    return utils.eval(test_file, output_file)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default=None)
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default=None)
    parser.add_option("--output", dest="conll_output", metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--bert", dest="bert_embedding", help="BERT embeddings", metavar="FILE")
    parser.add_option("--dev_bert", dest="dev_bert_embedding", help="dev BERT embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--extr_dim", type="int", dest="extr_dim", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=3000)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--ce", type="int", dest="ce", default=100)
    parser.add_option("--char_dim", type="int", dest="char_dim", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--t", type="int", dest="t", default=75000)
    parser.add_option("--min_count", type="int", dest="min_count", default=7)
    parser.add_option("--epoch", type="int", dest="epoch", default=5000)
    parser.add_option("--arc_mlp", type="int", dest="arc_mlp", default=600)
    parser.add_option("--label_mlp", type="int", dest="label_mlp", default=600)
    parser.add_option("--lr", type="float", dest="lr", default=1e-3)
    parser.add_option("--beta1", type="float", dest="beta1", default=0)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.95)
    parser.add_option("--interpolation_coef", type="float", dest="interpolation_coef", default=0.025)
    parser.add_option("--mtl_coef", type="float", dest="mtl_coef", default=0.975)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="leaky")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--rnn", type="int", dest="rnn", help='dimension of rnn in each direction', default=300)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--no_char", action="store_false", dest="use_char", default=True)
    parser.add_option("--no_pos", action="store_false", dest="use_pos", default=True)
    parser.add_option("--no_lemma", action="store_false", dest="use_lemma", default=True)
    parser.add_option("--no_syn_head_loss_bp", action="store_false", dest="syn_head_loss_bp", default=True)
    parser.add_option("--task_specific_recurrent_layer", action="store_true", dest="task_specific_recurrent_layer", default=False)
    parser.add_option("--add_bert", action="store_true", dest="add_bert_features", default=False)
    parser.add_option("--stop", type="int", dest="stop", default=10000)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)
    parser.add_option("--dynet-gpus", type="int", dest="dynet-gpus", default=1)
    parser.add_option("--dynet-gpu", action="store_true", dest="dynet-gpu", default=False,
                      help='Use GPU instead of cpu.')
    parser.add_option("--dynet-weight-decay", type="float", dest="dynet-weight-decay", default=3e-9)
    parser.add_option("--task", type="string", dest="task", help="options: syntax, sem, multi", default="syntax")
    parser.add_option("--sharing_mode", type="string", dest="sharing_mode", help="options: shared, separate, sum", default="shared")
    #dropout parameters
    parser.add_option("--input_emb_dropout", type="float", dest="input_emb_dropout", default=0.2)
    parser.add_option("--charlstm_dropout", type="float", dest="charlstm_dropout", default=0.33)
    parser.add_option("--bilstm_recur_dropout", type="float", dest="bilstm_recur_dropout", default=0.25)
    parser.add_option("--bilstm_ff_dropout", type="float", dest="bilstm_ff_dropout", default=0.45)
    parser.add_option("--arc_dropout", type="float", dest="arc_dropout", default=0.25)
    parser.add_option("--label_dropout", type="float", dest="label_dropout", default=0.33)

    (options, args) = parser.parse_args()
    print 'options', options
    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        #test time
        with open(options.params, 'r') as paramsfp:
            w2i, l2i, pos, dep_rels, sem_rels, chars, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        print 'stored options:', stored_opt
        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(pos, dep_rels, sem_rels, w2i, l2i, chars, stored_opt)
        parser.Load(options.model)
        ts = time.time()
        print 'loading buckets'
        test_buckets = [list()]
        test_data = list(utils.read_conll(open(options.conll_test, 'r')))
        for i, d in enumerate(test_data):
            test_buckets[0].append((i,d))
        print 'parsing'
        parse(parser, test_buckets, options.conll_test, options.conll_output, is_dev=False)
        te = time.time()
        print 'Finished predicting test.', te - ts, 'seconds.'

    else:
        print 'Preparing vocab'
        w2i, l2i, pos, dep_rels, sem_rels, chars = utils.vocab(options.conll_train, options.min_count)
        if not os.path.isdir(options.output): os.mkdir(options.output)
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((w2i, l2i, pos, dep_rels, sem_rels, chars, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(pos, dep_rels, sem_rels, w2i, l2i, chars, options)
        best_acc = -float('inf')
        t, epoch = 0, 1
        train_data = list(utils.read_conll(open(options.conll_train, 'r')))
        max_len = max([len(d) for d in train_data])
        min_len = min([len(d) for d in train_data])
        buckets = [list() for i in range(min_len, max_len + 1)]
        for i, d in enumerate(train_data):
            buckets[min(0, len(d) - min_len - 1)].append((i,d))
        buckets = [x for x in buckets if x != []]
        dev_buckets = [list()]
        if options.conll_dev:
            dev_data = list(utils.read_conll(open(options.conll_dev, 'r')))
            for j, d in enumerate(dev_data):
                dev_buckets[0].append((j,d))
        best_las = 0
        best_lf = 0
        no_improvement = 0
        while t <= options.t and epoch <= options.epoch:
            print 'Starting epoch', epoch, 'time:', time.ctime()
            mini_batches = utils.get_batches(buckets, parser, True)
            start, closs = time.time(), 0
            for i, minibatch in enumerate(mini_batches):
                if options.task == "syntax":
                    t, loss = parser.build_syntax_graph(minibatch, t)
                elif options.task == "sem":
                    t, loss = parser.build_semantic_graph(minibatch, t)
                elif options.task == "multi":
                    t, loss = parser.build_mtl_graph(minibatch, options.sharing_mode, t)
                else:
                    print 'unknown task option'
                    sys.exit(1)
                if parser.options.anneal:
                    decay_steps = min(1.0, float(t) / 50000)
                    lr = parser.options.lr * 0.75 ** decay_steps
                    parser.trainer.learning_rate = lr
                closs += loss
                if t % 10 == 0:
                    sys.stdout.write(
                        'overall progress:' + str(round(100 * float(t) / options.t, 2)) + '% current progress:' + str(
                            round(100 * float(i + 1) / len(mini_batches), 2)) + '% loss=' + str(
                            closs / 10) + ' time: ' + str(time.time() - start) + '\n')
                    if t % 100 == 0 and options.conll_dev:
                        uas, las, uf, lf = parse(parser, dev_buckets, options.conll_dev, options.output + '/dev.out', is_dev=True)

                        if options.task == 'syntax':
                            print 'current syntax accuracy', best_las, uas
                            if las > best_las:
                                best_las = las
                                print 'saving with', best_las, uas
                                parser.Save(options.output + '/model')
                                no_improvement = 0
                            else:
                                no_improvement += 1
                        elif options.task == 'sem':
                            print 'current semantic LF', best_lf, uf
                            if lf > best_lf:
                                best_lf = lf
                                print 'saving with', best_lf, lf
                                parser.Save(options.output + '/model')
                                no_improvement = 0
                            else:
                                no_improvement += 1
                        elif options.task == 'multi':
                            print 'current syntax accuracy', best_las, uas
                            print 'current semantic LF', best_lf, lf
                            if lf * las > best_lf * best_las:
                                best_lf = lf
                                best_las = las
                                print 'saving with', best_lf, lf, best_las, las
                                parser.Save(options.output + '/model')
                                no_improvement = 0
                            else:
                                no_improvement += 1

                    start, closs = time.time(), 0

                if no_improvement > options.stop:
                    print 'No improvements after', no_improvement, 'steps -> terminating.'
                    sys.exit(0)
            print 'current learning rate', parser.trainer.learning_rate, 't:', t
            epoch += 1

        if not options.conll_dev:
            print 'Saving default model without dev-tuning'
            parser.Save(options.output + '/model')
