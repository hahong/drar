"""Dropbox-based Archiving Utility
Author: Ha Hong (hahong84@gmail.com)
"""
import time
import os
import glob
import hashlib
import cPickle as pk
import sys
from dropbox import client, session, rest
from math import log

LSR_EXT = '.lst'
REMOTE_PATH = '/AR'
APP_IDENT_FILE = 'AR_app_ident.txt'
LARGE_TMP = '/home/hahong/teleport/space/tmp'
MAX_N_PER_DIR = 2000   # maximum number of files per folder in Dropbox
ELOG_EXT = '.elog.pkl'
MAP_EXT = '.map.txt'


# -- Main worker functions
def make_lsR(coll, newer=None, find_newer=True, t_slack=100):
    t0 = time.time()
    coll = coll.replace(os.sep, '')
    bn_lsR = '%s_lsR_' % coll
    fn_lsR_base = '%s%f' % (bn_lsR, t0)
    fn_lsR = fn_lsR_base + LSR_EXT
    ext = ''

    os.chdir(coll)
    # make sure we cd to coll and go back to ..
    try:
        if find_newer:
            lsR_ = sorted(glob.glob('%s*%s' % (bn_lsR, LSR_EXT)))
            if len(lsR_) > 0:
                newer = lsR_[-1]

        if newer is not None:
            ext += '-newer %s ' % newer

        # get list of files to backup
        os.system(r"find %s/ \( -type f -or -type l \) %s -exec 'ls'"
                ' --full-time -i {} + > %s' % (coll, ext, fn_lsR))
        # kludge way of retouching the lsR file's mtime just created
        # for the next incremental backup.
        os.system('touch -d "%s" %s' % (time.ctime(t0 - t_slack), fn_lsR))
    except Exception, e:
        print 'Failed:', e
        fn_lsR = None
    os.chdir('..')

    return coll + os.sep + fn_lsR, fn_lsR_base


def prepare_one_rec(coll, rec, arname=None, wd='tmp',
        i_inode=0, normalexit=0, large_tmp=LARGE_TMP):
    if not is_regularfile(rec):
        return False, '*** Not not regular file: ' + rec

    fn = rec[rec.index(coll + os.sep):]
    if not os.path.exists(coll + os.sep + fn):
        return False, '*** File not exists: ' + coll + os.sep + fn
    fsz = my_filesize(coll + os.sep + fn)

    if arname is None:
        arname = rec.split()[i_inode]
    elif arname == 'original':
        arname = os.path.basename(fn)
    spbase = wd + os.sep + arname + '.tar.'
    cspbase = coll + os.sep + spbase

    fullwd = coll + os.sep + wd
    my_makedirs(fullwd)
    if my_freespace(fullwd) < fsz and os.path.exists(large_tmp):
        if my_freespace(large_tmp) < fsz:
            # return False, '*** Not enough space: ' + fn
            print '*** Not enough space.  Need free space on:', large_tmp
            raw_input()
            if my_freespace(large_tmp) < fsz:
                return False, '*** Not enough space: ' + fn
        fullwd = large_tmp
        spbase = os.path.abspath(large_tmp) + os.sep + arname + '.tar.'
        cspbase = spbase

    # NOT USING COMPRESSION - too slow..
    # r = os.system('cd %s; tar czpf - "%s" | split -a 3 -d -b 200M'
    #         ' - "%s"' % (coll, fn, spbase))
    r = os.system('cd %s; tar cpf - "%s" | split -a 3 -d -b 200M'
            ' - "%s"' % (coll, fn, spbase))
    return r == normalexit, (cspbase, fn)


def do_incremental_backup(coll, elogext=ELOG_EXT, mapext=MAP_EXT,
        remote_path=REMOTE_PATH, rm_elog_on_succ=True):
    # -- collect lsR data and init dropbox space
    fn_lsR, bname = make_lsR(coll)
    fn_map = bname + mapext
    fn_elog = bname + elogext
    dstbase = remote_path + '/' + coll + '/' + bname

    recs = get_records(fn_lsR)
    if len(recs) == 0:
        print '*** No files to backup.  Exiting.'
        return

    # estimated number of depth required to hold all splits
    nfe = get_estimated_filenum(recs)
    nd = get_required_depth(nfe)
    nr = len(recs)

    print '* Starting archiving:', coll
    print '  - fn_map:  ', fn_map
    print '  - fn_elog: ', fn_elog
    print '  - dstbase: ', dstbase
    print '  - # files: ', nr
    print '  - # sp est:', nfe
    print '  - # dir d: ', nd
    print

    # setup dropbox connection
    app_key, app_secret = open(APP_IDENT_FILE).read().strip().split('|')
    sess = StoredSession(app_key, app_secret, 'dropbox')
    sess.load_creds()
    apicli = client.DropboxClient(sess)
    dbox_makedirs(apicli, dstbase)

    # -- pack one-by-one and upload files
    mf = open(fn_map, 'wt')
    ef = open(fn_elog, 'wb')
    ne = 0
    nf = -1
    for irec, rec in enumerate(recs):
        inf = None
        # compress the rec and split into small pieces
        pp_progress('At (%d%%): %s' % (100 * irec / nr, 'splitting...'))
        if inf is None:
            succ, inf = prepare_one_rec(coll, rec)
        if not succ:
            print '\n*** Error: prepare_one_rec():', inf
            my_dump({'func': 'prepare_one_rec', 'rec': rec,
                'irec': irec, 'inf': inf,
                'nf': nf, 'status': 'failed'}, ef)
            ne += 1
            continue

        spb, fn_src = inf
        my_dump({'func': 'prepare_one_rec', 'rec': rec,
            'irec': irec, 'nf': nf, 'status': 'ok'}, ef)
        pp_progress('At (%d%%): %s' % (100 * irec / nr, fn_src))

        splits = sorted(glob.glob(spb + '*'))
        for isp, sp in enumerate(splits):
            nf += 1   # increase regardless of success for safety
            coord = get_coord(nd, nf)[:nd]
            dst = (dstbase + '/' + 'd%04d/' * nd) % coord
            dbox_makedirs(apicli, dst)   # make sure there's holding dir
            dst += 'r%08d.tar.s%04d' % (irec, isp)

            pp_progress('At (%d%%): %s' % (100 * irec / nr,
                fn_src + ' -> ' + dst))
            succ, inf = dbox_upload(apicli, sp, dst)
            if not succ:
                print '\n*** Error: dbox_upload():', inf
                my_dump({'func': 'dbox_upload', 'rec': rec, 'spb': spb,
                    'sp': sp, 'dst': dst, 'fn_src': fn_src,
                    'inf': inf, 'nf': nf, 'status': 'failed'}, ef)
                ne += 1
                continue

            my_dump({'func': 'dbox_upload', 'rec': rec,
                'irec': irec, 'sp': sp, 'nf': nf, 'status': 'ok'}, ef)

            print >>mf, '%d\t%d\t%s\t%s\t%s\t%s' % \
                    (irec, nf, fn_src, sp, dst, inf)   # last one is hash
            mf.flush()

        my_dump({'func': 'rec_loop', 'rec': rec,
            'irec': irec, 'nf': nf, 'status': 'ok'}, ef)

    # -- cleanup
    mf.close()
    ef.close()

    print '\r' + ' ' * 90
    if ne > 0:
        print '*** There were %d errors.' % ne
        print '    Saved error logs: %s' % fn_elog
    else:
        dbox_upload(apicli, fn_map, dstbase + '/' + fn_map)
        if rm_elog_on_succ:
            my_unlink(fn_elog)
    print '* Finished:', coll


# -- Helper functions
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


def get_records(fn_lsR):
    L = open(fn_lsR).readlines()
    L = [e.strip() for e in L if is_regularfile(e)]
    return L


def get_estimated_filenum(recs, div=200 * 1024 * 1024, i_blks=5):
    n = 0
    for rec in recs:
        fs = float(rec.split()[i_blks])
        n0 = int(fs / div) + 1   # esimated splits for this rec
        n += n0
    return n


def get_required_depth(nf, nmax=MAX_N_PER_DIR):
    n = int(log(nf) / log(nmax)) + 1
    return n


def get_coord(nd, n, nmax=MAX_N_PER_DIR):
    if nd > 0:
        return (n / nmax ** nd,) + get_coord(nd - 1, n % nmax ** nd, nmax=nmax)
    assert n < nmax
    return (n,)


def is_regularfile(rec, i_perm=1):
    return rec.split()[i_perm].startswith('-')


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


def dbox_upload(apicli, src, dst, retry=5, delay503=60, delay=5, **kwargs):
    res = []
    for _ in xrange(retry):
        r = dbox_upload_once(apicli, src, dst, **kwargs)
        if r[0]:
            return r
        res.append(r[1])

        if '503' in r[1]:   # 503 Service Unavailable
            time.sleep(delay503)
        else:
            time.sleep(delay)
    return False, res


def dbox_upload_once(apicli, src, dst, halg=hashlib.sha224,
        tmpfnbase='tmp_', move=True, overwrite=True):
    tmpfn = None
    try:
        fsz = my_filesize(src)
        if dbox_df(apicli) <= fsz:
            print '*** Not enough dropbox space.  Need free space for:', src
            raw_input()
            if dbox_df(apicli) <= fsz:
                return False, '*** Not enough dropbox space: ' + src

        h0 = halg(open(src, 'rb').read()).hexdigest()
        # upload...
        apicli.put_file(dst, open(src, 'rb'), overwrite=overwrite)

        # ...and download to make sure everything is fine.
        tmpfn = tmpfnbase + str(time.time())
        tfp = open(tmpfn, 'wb')
        f, _ = apicli.get_file_and_metadata(dst)
        tfp.write(f.read())
        tfp.close()
        f.close()

        h1 = halg(open(tmpfn, 'rb').read()).hexdigest()
        my_unlink(tmpfn)
        if h0 != h1:
            apicli.file_delete(dst)
            raise ValueError('Not matching hashes: %s != %s' % (h0, h1))

        if move:
            my_unlink(src)

    except Exception, e:
        if type(tmpfn) is str:
            my_unlink(tmpfn)
        return False, 'Failed: ' + str(e)

    return True, h0


def dbox_makedirs(apicli, path):
    px = path.split('/')
    n = len(px)

    for i in xrange(1, n):
        path_ = '/'.join(px[:i + 1])
        if dbox_exists(apicli, path_):
            continue
        apicli.file_create_folder(path_)

    assert dbox_exists(apicli, path)   # assure the path is made


def dbox_exists(apicli, path, info=False):
    try:
        res = apicli.metadata(path)
        if res.get('is_deleted'):
            return False

        if info:
            return res
        return True
    except rest.ErrorResponse:
        pass
    return False


def dbox_df(apicli):
    try:
        inf = apicli.account_info()
        return inf['quota_info']['quota'] - inf['quota_info']['normal'] - \
                inf['quota_info']['shared']
    except rest.ErrorResponse:
        pass
    return -1


def main_cli(args):
    do_incremental_backup(args[0])


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
