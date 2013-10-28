"""Dropbox-based Archiving Utility
Author: Ha Hong (hahong84@gmail.com)
"""
import time
import os
import glob
import hashlib
import cPickle as pk
import sys
import signal
from dropbox import client, session, rest
from math import log, log10
# from functools import wraps

LSR_EXT = '.lsR.lst'
LSR_EXT_FULL = '.full.lsR.lst'
REMOTE_PATH = '/AR'
APP_IDENT_FILE = 'AR_app_ident.txt'
LARGE_TMP = '/home/hahong/teleport/space/tmp'
MAX_N_PER_DIR = 2000   # maximum number of files per folder in Dropbox
LOG_EXT = '.log.pkl'
MAP_EXT = '.map.txt'
TIMEOUT_SHORT = 45
TIMEOUT_LONG = 60 * 10


# -- Main worker functions
def make_lsR(coll, newer=None, find_newer=True, t_slack=100,
        fn_lsR_base=None, lsR_ext=LSR_EXT, incl_all=False):
    t0 = time.time()
    coll = coll.replace(os.sep, '')
    bn_lsR = '%s_' % coll
    if fn_lsR_base is None:
        fn_lsR_base = '%s%f' % (bn_lsR, t0)
    fn_lsR = fn_lsR_base + lsR_ext
    if not incl_all:
        opts = r'\( -type f -or -type l \) '
        opts2 = ''
    else:
        opts = ''
        opts2 = '-dF'

    os.chdir(coll)
    # make sure we cd to coll and go back to ..
    try:
        if find_newer:
            lsR_ = sorted(glob.glob('%s*%s' % (bn_lsR, lsR_ext)))
            if len(lsR_) > 0:
                newer = lsR_[-1]

        if newer is not None:
            opts += '-newer %s ' % newer

        # get list of files to backup
        os.system(r"find %s/ %s -exec 'ls'"
                ' --full-time -i %s {} + > %s' % (coll, opts, opts2, fn_lsR))
        # kludge way of retouching the lsR file's mtime just created
        # for the next incremental backup.
        os.system('touch -d "%s" %s' % (time.ctime(t0 - t_slack), fn_lsR))
    except Exception, e:
        print 'Failed:', e
        fn_lsR = None
    os.chdir('..')

    return coll + os.sep + fn_lsR, fn_lsR_base


def prepare_one_rec(coll, recdesc, arname=None, wd='tmp', recs=None,
        i_inode=0, normalexit=0, large_tmp=LARGE_TMP,
        skip_hard_work=False, nicer_fn=True,
        sp_patt='.tar.lzo.sp', i_sp_patt_cutoff=-3):
    rec, irec, nrec = recdesc
    is_regf, ftype = is_regularfile(rec, full=True)
    if not is_regf:
        return False, '*** Not not regular file: ' + rec

    fn = rec[rec.index(coll + os.sep):]
    fn = get_regular_filename(fn, ftype)
    pth = coll + os.sep + fn
    # os.path.islink is needed to allow broken symlinks
    if not os.path.exists(pth) and not os.path.islink(pth):
        return False, '*** File not exists: ' + pth
    fsz = my_filesize(pth)

    # -- determine file name in the archive "aname"
    if arname is None:
        # gives filename based on the record number
        arname = 'f' + get_padded_numstr(irec, nrec) + '_' + \
                get_dboxsafe_filename(os.path.basename(fn))
    elif arname == '__inode__':
        # gives filename based on the inode
        arname = rec.split()[i_inode] + '_' + \
                get_dboxsafe_filename(os.path.basename(fn))
    elif arname == '__original__':
        arname = os.path.basename(fn)

    # process hardlinks if recs is present
    hardlink = False
    if recs is not None:
        inode0 = int(rec.split()[i_inode])  # this record's inode
        for irec1, rec1 in recs[:irec]:
            inode1 = int(rec1.split()[i_inode])
            if inode1 != inode0:
                continue
            hardlink = True
            fn1 = rec1[rec1.index(coll + os.sep):]
            arname += '__hardlink_to__f%s.txt' % get_padded_numstr(irec1, nrec)

    spbase = wd + os.sep + arname + (sp_patt if not hardlink else '')
    cspbase = coll + os.sep + spbase
    if skip_hard_work:
        cspbase = cspbase[:i_sp_patt_cutoff]
        print '* Skip actual file splitting:', cspbase
        return True, (cspbase, fn)

    # -- if hardlink, then just write the path to already transferred data.
    if hardlink:
        open(cspbase, 'wt').write(fn1 + '\n->\n' + fn)
        return True, (cspbase, fn)

    # -- split the file if needed
    fullwd = coll + os.sep + wd
    my_makedirs(fullwd)
    if my_freespace(fullwd) < fsz and os.path.exists(large_tmp):
        if my_freespace(large_tmp) < fsz:
            # return False, '*** Not enough space: ' + fn
            set_alarm(0)
            print '*** Not enough space.  Need free space on:', large_tmp
            print '*** User action required.  Press enter after done.'
            raw_input()
            if my_freespace(large_tmp) < fsz:
                return False, '*** Not enough space: ' + fn
        fullwd = large_tmp
        spbase = os.path.abspath(large_tmp) + os.sep + arname + sp_patt
        cspbase = spbase

    #r = os.system('cd %s; tar --lzma -cpf - "%s" | split -a 9 -d -b 200M'
    #r = os.system('cd %s; tar zcpf - "%s" | split -a 9 -d -b 200M'
    r = os.system('cd %s; tar --lzop -cpf - "%s" | split -a 9 -d -b 200M'
            ' - "%s"' % (coll, fn, spbase))
    if nicer_fn:
        splits = sorted(glob.glob(cspbase + '*'))
        ns = len(splits)
        if ns > 1:
            sptot = '.tot%d' % ns
            for sp in splits:
                spi = int(sp.split(cspbase)[-1])
                # make it 1-based for human readability
                spi = get_padded_numstr(spi + 1, ns)
                spnew = cspbase + spi
                os.rename(sp, spnew + sptot)
        elif ns == 1:
            fnnew = '.'.join(splits[0].split('.')[:-1])
            os.rename(splits[0], fnnew)
            cspbase = fnnew
        else:
            return False, '*** Cannot find files: ' + cspbase

    return r == normalexit, (cspbase, fn)


def do_incremental_backup(coll, logext=LOG_EXT, mapext=MAP_EXT,
        remote_path=REMOTE_PATH, recover=None, force_continue=False):
    # -- collect lsR data and init dropbox space
    if recover is None:
        fn_lsR, bname = make_lsR(coll)
        make_lsR(coll, find_newer=False, incl_all=True,
                fn_lsR_base=bname, lsR_ext=LSR_EXT_FULL)
    else:
        fn_lsR = recover['fn_lsR']
        bname = recover['bname']
    fn_map = bname + mapext
    fn_log = bname + logext
    dstbase = remote_path + '/' + coll + '/' + bname

    recs = get_records(fn_lsR)
    # if len(recs) == 0:
    #    print '*** No files to backup.  Exiting.'
    #    return

    # estimated number of depth required to hold all splits
    nfe = get_estimated_filenum(recs)
    nd = get_required_depth(nfe)
    nr = len(recs)

    print '* Starting archiving:', coll
    print '  - fn_map:  ', fn_map
    print '  - fn_log:  ', fn_log
    print '  - dstbase: ', dstbase
    print '  - # files: ', nr
    print '  - # sp est:', nfe
    print '  - # dir d: ', nd
    print

    # setup dropbox connection
    apicli, _ = make_conn()
    dbox_makedirs(apicli, dstbase)

    # -- pack one-by-one and upload files
    mf = open(fn_map, 'at')
    ef = open(fn_log, 'ab')
    ne = 0
    nf = 0 if recover is None else recover['nf_base']
    ib = 0 if recover is None else recover['ib']
    recinf_recover = None if recover is None else recover['recinf']
    spinf_recover = None if recover is None else recover['spinf']
    confirm_once = False if recover is None else True
    skip_hard_work_once = False

    if recinf_recover is not None:
        skip_hard_work_once = recinf_recover.get('skip_hard_work_once', False)
        recinf_recover = None

    for irec, rec in recs[ib:]:
        # compress the rec and split into small pieces
        pp_progress('At (%d%%): %s' % (100 * irec / nr, 'splitting...'))
        set_alarm(0)
        succ, inf = prepare_one_rec(coll, (rec, irec, nr),
                skip_hard_work=skip_hard_work_once, recs=recs)
        skip_hard_work_once = False

        if not succ:
            print '\n*** Error: prepare_one_rec():', inf
            my_dump({'func': 'prepare_one_rec', 'rec': rec,
                'irec': irec, 'inf': inf,
                'nf': nf, 'status': 'failed'}, ef)
            ne += 1
            if force_continue:
                continue
            print '*** Aborting.'
            break

        # get info and journal stuffs
        spb, fn_src = inf
        my_dump({'func': 'prepare_one_rec', 'rec': rec,
            'irec': irec, 'nf': nf, 'status': 'ok'}, ef)
        pp_progress('At (%d%%): %s' % (100 * irec / nr, fn_src))

        # get splits and handle recovery info if needed
        splits = sorted(glob.glob(spb + '*'))
        if spinf_recover is not None:
            n_sp_done = spinf_recover['n_sp_done']
            nf += n_sp_done
            spinf_recover = None

        for sp in splits:
            coord = get_coord(nd, nf)[:nd]
            dst = (dstbase + '/' + 'd%04d/' * nd) % coord
            dbox_makedirs(apicli, dst)   # make sure there's holding dir
            dst += os.path.basename(sp)
            dbox_overwrite_check(apicli, dst)
            if confirm_once:
                print '\n* Confirmation required:'
                print '   - nf: ', nf
                print '   - sp: ', sp
                print '   - dst:', dst
                print '*** Press ^C to halt.  Otherwise, press enter.'
                raw_input()
                confirm_once = False

            pp_progress('At (%d%%): %s' % (100 * irec / nr,
                fn_src + ' -> ' + dst))
            succ, inf = dbox_upload(apicli, sp, dst, retry=100000)

            if succ:
                my_dump({'func': 'dbox_upload', 'rec': rec,
                    'irec': irec, 'sp': sp, 'nf': nf, 'status': 'ok'}, ef)

                print >>mf, '%d\t%d\t%s\t%s\t%s\t%s' % \
                        (irec, nf, fn_src, sp, dst, inf)   # last one is hash
                mf.flush()
            else:
                print '\n*** Error: dbox_upload():', inf
                my_dump({'func': 'dbox_upload', 'irec': irec, 'rec': rec,
                    'spb': spb, 'sp': sp, 'dst': dst, 'fn_src': fn_src,
                    'inf': inf, 'nf': nf, 'status': 'failed'}, ef)
                ne += 1

            # increase regardless of success
            nf += 1

        my_dump({'func': 'rec_loop', 'rec': rec,
            'irec': irec, 'nf': nf, 'status': 'ok'}, ef)

    # -- cleanup
    set_alarm(0)
    mf.close()
    ef.close()

    print '\r' + ' ' * 90
    if ne > 0:
        print '*** There were %d errors logged as: %s' % (ne, fn_log)
    else:
        fn_lsR_full = fn_lsR.replace(LSR_EXT, LSR_EXT_FULL)
        dbox_upload(apicli, fn_map, dstbase + '/' + os.path.basename(fn_map))
        dbox_upload(apicli, fn_log, dstbase + '/' + os.path.basename(fn_log))
        dbox_upload(apicli, fn_lsR, dstbase + '/' + os.path.basename(fn_lsR),
                move=False)
        if os.path.exists(fn_lsR_full):
            dbox_upload(apicli, fn_lsR_full, dstbase + '/' +
                    os.path.basename(fn_lsR_full),
                    move=False)
    print '* Finished:', coll


def do_recovery(coll, fn_lsR, logfn, lsR_ext=LSR_EXT):
    nf_base = 0
    nf_last = 0
    ib = 0
    recinf = None
    spinf = None

    # -- load logs
    fp = open(logfn)
    try:
        while True:
            l = pk.load(fp)
            if l['func'] == 'prepare_one_rec':
                nf_base = l['nf']
            nf_last = l['nf']
            if 'irec' in l:
                ib = l['irec']
    except:
        pass

    n_sp_done = nf_last - nf_base
    log_last = l   # last entry of logs

    # assert log_last['status'] == 'failed'   # not needed
    print '* Last record:'
    print log_last
    print '*** Press ^C to halt.  Otherwise, press enter.'
    raw_input()

    # -- make recovery point
    if log_last['func'] == 'prepare_one_rec' and \
            log_last['status'] == 'failed':
        pass
    elif log_last['func'] == 'prepare_one_rec' and \
            log_last['status'] == 'ok':
        recinf = {'skip_hard_work_once': True}
    elif log_last['func'] == 'dbox_upload' and \
            log_last['status'] == 'failed':
        recinf = {'skip_hard_work_once': True}
        spinf = {'n_sp_done': n_sp_done}
    elif log_last['func'] == 'dbox_upload' and \
            log_last['status'] == 'ok':
        recinf = {'skip_hard_work_once': True}
        spinf = {'n_sp_done': n_sp_done + 1}
    else:
        raise ValueError('Cannot understand recovery point.')

    recover = {}
    recover['fn_lsR'] = fn_lsR
    recover['bname'] = os.path.basename(fn_lsR).replace(lsR_ext, '')
    recover['nf_base'] = nf_base
    recover['ib'] = ib
    recover['recinf'] = recinf
    recover['spinf'] = spinf

    do_incremental_backup(coll, recover=recover)


# -- Helper functions
class TimeoutError(Exception):
    pass


def _handle_timeout(signum, frame):
    # error_message = os.strerror(errno.ETIME)
    signal.alarm(0)
    raise TimeoutError('timeout')


def set_alarm(t=TIMEOUT_SHORT):
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(t)


# from: http://stackoverflow.com/questions/2281850/ ...
#       timeout-function-if-it-takes-too-long-to-finish
# def timeout(seconds=TIMEOUT_SHORT, default=None):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             set_alarm(seconds)
#             try:
#                 result = func(*args, **kwargs)
#             except TimeoutError:
#                 result = default
#             return result
#         return wraps(func)(wrapper)
#     return decorator


def pp_progress(s, l=90, c=20, el=' ... '):
    n = len(s)
    if n > l:
        b = l - c + len(el)
        s = s[:c] + el + s[-b:]
    n = len(s)
    ne = pp_progress.nprev - n   # not need to worry about negative
    print '\r' + s + ' ' * ne + '\r',
    sys.stdout.flush()
    pp_progress.nprev = n
pp_progress.nprev = 0


def make_conn(app_ident_file=APP_IDENT_FILE, ):
    app_key, app_secret = open(app_ident_file).read().strip().split('|')
    sess = StoredSession(app_key, app_secret, 'dropbox')
    sess.load_creds()
    apicli = client.DropboxClient(sess)
    return apicli, sess


def get_records(fn_lsR):
    L = open(fn_lsR).readlines()
    L = [e.strip() for e in L if is_regularfile(e)]
    L = [(i, e) for i, e in enumerate(L)]
    return L


def get_estimated_filenum(recs, div=200 * 1024 * 1024, i_blks=5):
    n = 0
    for irec, rec in recs:
        fs = float(rec.split()[i_blks])
        n0 = int(fs / div) + 1   # esimated splits for this rec
        n += n0
    return n


def get_required_depth(nf, nmax=MAX_N_PER_DIR):
    if nf <= 0:
        return 1
    n = int(log(nf) / log(nmax)) + 1
    return n


def get_coord(nd, n, nmax=MAX_N_PER_DIR):
    if nd > 0:
        return (n / nmax ** nd,) + get_coord(nd - 1, n % nmax ** nd, nmax=nmax)
    assert n < nmax
    return (n,)


def is_regularfile(rec, i_perm=1, permit_symlinks=True, full=False):
    permits = ('-',)
    if permit_symlinks:
        permits += ('l',)

    stperm = rec.split()[i_perm]
    if full:
        return stperm.startswith(permits), stperm[0]
    return stperm.startswith(permits)


def get_regular_filename(fn, ftype):
    if ftype == '-':
        return fn
    if ftype == 'l':  # symlinks
        return fn.split(' -> ')[0]


def my_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def my_freespace(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize


def my_unlink(fn):
    if not os.path.exists(fn):
        return
    os.unlink(fn)


def my_filesize(fn):
    if not os.path.exists(fn):
        return -1
    return os.stat(fn).st_size


def my_dump(obj, fp):
    pk.dump(obj, fp)
    fp.flush()


def get_padded_numstr(n, ntotal):
    npad = int(log10(ntotal) + 1)
    spad = '%%0%dd' % npad
    return spad % n


def get_dboxsafe_filename(fn, allowed=['_', '-', '.']):
    return ''.join(e if e.isalnum() or e in allowed else '_' for e in str(fn))


def dbox_overwrite_check(apicli, dst, retry=5):
    if not dbox_exists(apicli, dst):
        return
    set_alarm(0)
    print '\n*** File exists:', dst
    print '*** User action required.  Press enter after done.'
    raw_input()
    if retry > 0:
        dbox_overwrite_check(apicli, dst, retry=retry - 1)


def dbox_upload(apicli, src, dst, retry=10, delay_long=120, delay=30,
        **kwargs):
    res = []
    while retry > 0:
        r = dbox_upload_once(apicli, src, dst, **kwargs)
        if r[0]:
            return r
        res.append(r[1])

        if '503' in r[1]:   # 503 Service Unavailable
            time.sleep(delay_long)
        else:
            time.sleep(delay)
        retry -= 1

    return False, res


def dbox_upload_once(apicli, src, dst, halg=hashlib.sha224,
        tmpfnbase='tmp_', move=True, overwrite=True):
    tmpfn = None
    try:
        fsz = my_filesize(src)
        if dbox_df(apicli) <= fsz:
            set_alarm(0)
            print '\n*** Not enough dropbox space.  Need %db for: %s' % \
                    (fsz, src)
            print '*** User action required.  Press enter after done.'
            raw_input()
            if dbox_df(apicli) <= fsz:
                return False, '*** Not enough dropbox space: ' + src

        set_alarm(TIMEOUT_LONG)
        h0 = halg(open(src, 'rb').read()).hexdigest()
        # upload...
        apicli.put_file(dst, open(src, 'rb'), overwrite=overwrite)

        # ...and download to make sure everything is fine.
        set_alarm(TIMEOUT_LONG)
        tmpfn = tmpfnbase + str(time.time())
        tfp = open(tmpfn, 'wb')
        f, _ = apicli.get_file_and_metadata(dst)
        tfp.write(f.read())
        tfp.close()
        f.close()

        # test downloaded one
        h1 = halg(open(tmpfn, 'rb').read()).hexdigest()
        my_unlink(tmpfn)
        if h0 != h1:
            apicli.file_delete(dst)
            raise ValueError('Not matching hashes: %s != %s' % (h0, h1))
        if move:
            my_unlink(src)

    except Exception, e:
        set_alarm(0)
        if type(tmpfn) is str:
            my_unlink(tmpfn)
        return False, 'Failed: ' + str(e)

    set_alarm(0)
    return True, h0


def dbox_makedirs(apicli, path, retry=5, delay=5):
    px = path.split('/')
    n = len(px)

    for i in xrange(1, n):
        path_ = '/'.join(px[:i + 1])
        if dbox_exists(apicli, path_):
            continue
        for _ in xrange(retry):
            try:
                set_alarm()
                apicli.file_create_folder(path_)
                set_alarm(0)
                break
            except Exception:
                time.sleep(delay)

    assert dbox_exists(apicli, path)   # assure the path is made


def dbox_exists(apicli, path, info=False, retry=5, delay=5):
    try:
        set_alarm()
        res = apicli.metadata(path)
        set_alarm(0)
        if res.get('is_deleted'):
            return False

        if info:
            return res
        return True
    except Exception, e:
        set_alarm(0)
        # 404: not exists
        if type(e) is rest.ErrorResponse and e.status == 404:
            return False
        # print '\n*** Error: dbox_exists():', e
        if retry > 0:
            time.sleep(delay)
            return dbox_exists(apicli, path, info=info,
                    retry=retry - 1, delay=delay)
    return False


def dbox_df(apicli, retry=5, delay=5):
    try:
        set_alarm()
        inf = apicli.account_info()
        set_alarm(0)
        return inf['quota_info']['quota'] - inf['quota_info']['normal'] - \
                inf['quota_info']['shared']
    except Exception:
        set_alarm(0)
        # print '\n*** Error: dbox_df():', e
        if retry > 0:
            time.sleep(delay)
            return dbox_df(apicli, retry=retry - 1, delay=delay)
    return -1


def main_cli(args):
    if len(args) == 1:
        do_incremental_backup(args[0])
    elif len(args) == 3:
        coll, fn_lsR, logfn = args
        do_recovery(coll, fn_lsR, logfn)
    else:
        print 'Arguments not understood.'


# -- Copied from dropbox example's cli_client.py
class StoredSession(session.DropboxSession):
    """a wrapper around DropboxSession that stores a token to a file on disk"""
    TOKEN_FILE = "AR_token_store.txt"

    def load_creds(self):
        try:
            stored_creds = open(self.TOKEN_FILE).read()
            self.set_token(*stored_creds.split('|'))
            print "[loaded access token]"
        except IOError:
            self.link()

    def write_creds(self, token):
        f = open(self.TOKEN_FILE, 'w')
        f.write("|".join([token.key, token.secret]))
        f.close()

    def delete_creds(self):
        my_unlink(self.TOKEN_FILE)

    def link(self):
        request_token = self.obtain_request_token()
        url = self.build_authorize_url(request_token)
        print "url:", url
        print "Please authorize in the browser. After done, press enter."
        raw_input()

        self.obtain_access_token(request_token)
        self.write_creds(self.token)

    def unlink(self):
        self.delete_creds()
        session.DropboxSession.unlink(self)
