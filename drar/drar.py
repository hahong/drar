"""Dropbox-based Archiving Utility
Author: Ha Hong (hahong84@gmail.com)
"""
import time
import os
import glob
from dropbox import client, session, rest

LSR_EXT = '.lst'
REMOTE_PATH = '/AR'
APP_IDENT_FILE = 'AR_app_ident.txt'


# -- Main worker functions
def make_lsR(coll, newer=None, find_newer=True, t_slack=100):
    t0 = time.time()
    coll = coll.replace(os.sep, '')
    bn_lsR = '%s_lsR_' % coll
    fn_lsR = '%s%f%s' % (bn_lsR, t0, LSR_EXT)
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
    os.chdir('..')

    return fn_lsR


def compress_one_rec(coll, rec, arname=None, wd='tmp',
        i_perm=1, i_inode=0, normalexit=0):
    if not rec.split()[i_perm].startswith('-'):
        return True, None   # not regular file

    fn = rec[rec.index(coll + os.sep):]
    if not os.path.exists(coll + os.sep + fn):
        return False, None

    if arname is None:
        arname = rec.split()[i_inode]
    elif arname == 'original':
        arname = os.path.basename(fn)
    spbase = wd + os.sep + arname + '.tgz.'

    fullwd = coll + os.sep + wd
    if not os.path.exists(fullwd):
        os.makedirs(fullwd)

    r = os.system('cd %s; tar czpf - "%s" | split -a 3 -d -b 200M'
            ' - "%s"' % (coll, fn, spbase))
    return r == normalexit, spbase


def do_incremental_backup(coll):
    app_key, app_secret = open(APP_IDENT_FILE).read().strip().split('|')
    sess = StoredSession(app_key, app_secret, 'dropbox')
    sess.load_creds()
    apicli = client.DropboxClient(sess)

    dbox_makedirs(apicli, '/AR/4/5')


# -- Helper functions
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


def main_cli(args):
    pass


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
        os.unlink(self.TOKEN_FILE)

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
