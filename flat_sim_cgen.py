import nlsymb

#from nlsymb import np
#from nlsymb import sym as syp
import nlsymb.sys as sys

#S = syp.Symbol

if __name__=="__main__":
    # create symbolic system
    s = sys.FlatFloor2D(k=3)

    x = s.x
    u = s.u
    phi = x[s.si]

    # plus terms, when phi > 0
    fp = s._fplus.expr
    dfpdx = s._dfxp.expr
    dfpdu = s._dfup.expr

    # plus terms, when phi < 0
    fm = s._fmins.expr
    dfmdx = s._dfxm.expr
    dfmdu = s._dfum.expr
    
    print "made symbolic system"
