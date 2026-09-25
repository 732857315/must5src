/* Freestanding bounded Gomoku search. No C runtime, network or file access.
 * Explicit tactics prove outcomes; candidate pruning never proves a loss.
 * Host-owned context keeps recursive scratch storage off the Windows stack. */
#define MAX_CELLS 4096
#define MAX_SEGMENTS (4*MAX_CELLS)
#define MAX_PLY 64
#define MAX_WIDTH 64
#define MATE 100000000
#define UNKNOWN 2
#define TT_SIZE 32768
#if defined(__wasm__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT __declspec(dllexport)
#endif
typedef unsigned char u8;
typedef unsigned long long u64;
typedef long long i64;
int _fltused = 0;
void *memset(void *ptr, int value, __SIZE_TYPE__ size) { volatile u8 *p=(volatile u8*)ptr; while(size--) *p++=(u8)value; return ptr; }
void *memcpy(void *dst, const void *src, __SIZE_TYPE__ size) { u8 *d=(u8*)dst; const u8 *s=(const u8*)src; while(size--) *d++=*s++; return dst; }
typedef struct { int move; i64 order; } Candidate;
typedef struct {
    int score[2][MAX_CELLS];
    u8 wins[2][MAX_CELLS], upgrades[2][MAX_CELLS], frontier[MAX_CELLS];
    int win_count[2], empty_count, evaluation[2];
    Candidate choices[MAX_WIDTH];
} Work;
typedef int (*StopCallback)(void);
typedef struct {
    u64 key;
    int score,move;
    unsigned int generation;
    short depth;
    u8 flag;
} Transposition;
typedef struct {
    unsigned short lines5[MAX_CELLS][20],lines6[MAX_CELLS][24];
    u8 count5[MAX_CELLS],count6[MAX_CELLS],neighbors[MAX_CELLS];
    u8 wins[2][MAX_CELLS],upgrades[2][MAX_CELLS];
    int score[2][MAX_CELLS],evaluation[2],empty_count,stones;
} Incremental;
typedef struct {
    u8 board[MAX_CELLS]; double priors[MAX_CELLS];
    int segments5[MAX_SEGMENTS][5], segments6[MAX_SEGMENTS][6];
    int n5,n6,rows,cols,cells,root_side,width,max_depth,max_nodes,nodes,aborted;
    StopCallback stop;
    Work work[MAX_PLY];
    Incremental position;
    Transposition table[TT_SIZE];
    u64 hash;
    unsigned int generation;
    int table_hits;
} Context;
typedef struct { int score, proof; } Result;
typedef struct { int move,nodes,completed_depth,score,proof,budget_exhausted,status; } Output;
static const int WEIGHTS[6]={0,1,8,64,1024,1000000};
static const int DR[4]={0,1,1,1}, DC[4]={1,0,1,-1};
static int absolute(int value) { return value<0?-value:value; }
static int check(Context *c) {
    if(c->aborted) return 0;
    if(c->nodes>=c->max_nodes || ((c->nodes&31)==0 && c->stop && c->stop())) {c->aborted=1; return 0;}
    return 1;
}
static int won_at(Context *c,int move,int side) {
    int row=move/c->cols,col=move%c->cols;
    for(int d=0;d<4;d++) {
        int count=1;
        for(int sign=-1;sign<=1;sign+=2) for(int step=1;step<5;step++) {
            int r=row+sign*step*DR[d],q=col+sign*step*DC[d];
            if(r<0||q<0||r>=c->rows||q>=c->cols||c->board[r*c->cols+q]!=side) break;
            count++;
        }
        if(count>=5) return 1;
    }
    return 0;
}
static void analyse_full(Context *c,Work *w) {
    w->win_count[0]=w->win_count[1]=w->evaluation[0]=w->evaluation[1]=w->empty_count=0;
    for(int i=0;i<c->cells;i++) {
        w->score[0][i]=w->score[1][i]=0;
        w->wins[0][i]=w->wins[1][i]=w->upgrades[0][i]=w->upgrades[1][i]=w->frontier[i]=0;
        w->empty_count+=c->board[i]==0;
    }
    for(int k=0;k<c->n5;k++) {
        int *line=c->segments5[k],counts[4]={0,0,0,0};
        for(int i=0;i<5;i++) counts[c->board[line[i]]]++;
        if(counts[3]) continue;
        for(int s=0;s<2;s++) {
            if(counts[2-s]) continue;
            int n=counts[s+1];
            if(n>=5) continue;
            w->evaluation[s]+=WEIGHTS[n];
            int delta=WEIGHTS[n+1]-WEIGHTS[n];
            for(int i=0;i<5;i++) if(c->board[line[i]]==0) {
                int point=line[i]; w->score[s][point]+=delta;
                if(n==4 && !w->wins[s][point]) {w->wins[s][point]=1; w->win_count[s]++;}
            }
        }
    }
    for(int k=0;k<c->n6;k++) {
        int *line=c->segments6[k];
        if(c->board[line[0]]||c->board[line[5]]) continue;
        int counts[4]={0,0,0,0};
        for(int i=1;i<5;i++) counts[c->board[line[i]]]++;
        if(counts[3]) continue;
        for(int s=0;s<2;s++) if(!counts[2-s]) {
            int n=counts[s+1];
            if(n==3) {
                w->evaluation[s]+=160;
                for(int i=1;i<5;i++) if(!c->board[line[i]]) w->upgrades[s][line[i]]=1;
            } else if(n==2) {
                w->evaluation[s]+=12;
                for(int i=1;i<5;i++) if(!c->board[line[i]]) w->score[s][line[i]]+=240;
            }
        }
    }
    int any=0;
    for(int i=0;i<c->cells;i++) if(c->board[i]==1||c->board[i]==2) {
        any=1; int r=i/c->cols,q=i%c->cols;
        for(int dr=-2;dr<=2;dr++) for(int dc=-2;dc<=2;dc++) {
            int rr=r+dr,cc=q+dc;
            if(rr>=0&&rr<c->rows&&cc>=0&&cc<c->cols&&!c->board[rr*c->cols+cc]) w->frontier[rr*c->cols+cc]=1;
        }
    }
    if(!any) {
        int center=(c->rows/2)*c->cols+c->cols/2;
        if(!c->board[center]) w->frontier[center]=1;
        else for(int i=0;i<c->cells;i++) if(!c->board[i]) w->frontier[i]=1;
    }
    /* Fixed forbidden cells may isolate every stone from distant legal cells.
     * An empty frontier is not a full board and must never yield zero choices. */
    int nearby=0;
    for(int i=0;i<c->cells;i++) nearby|=w->frontier[i];
    if(!nearby) for(int i=0;i<c->cells;i++) if(!c->board[i]) w->frontier[i]=1;
}
/* Update only the 5/6-cell lines crossing a changed point. Keep the original
 * full scan above as an independent oracle for incremental-state regressions. */
static u64 piece_key(int point,int side) {
    u64 x=(u64)(point*4+side+1)+0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
static void update5(Context *c,int k,int sign) {
    Incremental *p=&c->position;
    int *line=c->segments5[k],counts[4]={0,0,0,0};
    for(int i=0;i<5;i++) counts[c->board[line[i]]]++;
    if(counts[3]) return;
    for(int s=0;s<2;s++) if(!counts[2-s]&&counts[s+1]<5) {
        int n=counts[s+1],delta=WEIGHTS[n+1]-WEIGHTS[n];
        p->evaluation[s]+=sign*WEIGHTS[n];
        for(int i=0;i<5;i++) if(!c->board[line[i]]) {
            p->score[s][line[i]]+=sign*delta;
            if(n==4) p->wins[s][line[i]]+=sign;
        }
    }
}
static void update6(Context *c,int k,int sign) {
    Incremental *p=&c->position;
    int *line=c->segments6[k],counts[4]={0,0,0,0};
    if(c->board[line[0]]||c->board[line[5]]) return;
    for(int i=1;i<5;i++) counts[c->board[line[i]]]++;
    if(counts[3]) return;
    for(int s=0;s<2;s++) if(!counts[2-s]) {
        int n=counts[s+1];
        if(n==3) {
            p->evaluation[s]+=sign*160;
            for(int i=1;i<5;i++) if(!c->board[line[i]]) p->upgrades[s][line[i]]+=sign;
        } else if(n==2) {
            p->evaluation[s]+=sign*12;
            for(int i=1;i<5;i++) if(!c->board[line[i]]) p->score[s][line[i]]+=sign*240;
        }
    }
}
static void update_neighbors(Context *c,int point,int sign) {
    int r=point/c->cols,q=point%c->cols;
    c->position.stones+=sign;
    for(int dr=-2;dr<=2;dr++) for(int dc=-2;dc<=2;dc++) {
        int rr=r+dr,cc=q+dc;
        if(rr>=0&&rr<c->rows&&cc>=0&&cc<c->cols) c->position.neighbors[rr*c->cols+cc]+=sign;
    }
}
static void initialize_position(Context *c) {
    Incremental *p=&c->position;
    p->evaluation[0]=p->evaluation[1]=p->empty_count=p->stones=0;
    c->hash=piece_key(c->rows*65+c->cols,0);
    for(int i=0;i<c->cells;i++) {
        p->count5[i]=p->count6[i]=p->neighbors[i]=0;
        for(int s=0;s<2;s++) p->score[s][i]=p->wins[s][i]=p->upgrades[s][i]=0;
        p->empty_count+=!c->board[i];
        if(c->board[i]) c->hash^=piece_key(i,c->board[i]);
    }
    for(int i=0;i<c->cells;i++) if(c->board[i]==1||c->board[i]==2) update_neighbors(c,i,1);
    for(int k=0;k<c->n5;k++) {
        for(int j=0;j<5;j++) {int i=c->segments5[k][j];p->lines5[i][p->count5[i]++]=(unsigned short)k;}
        update5(c,k,1);
    }
    for(int k=0;k<c->n6;k++) {
        for(int j=0;j<6;j++) {int i=c->segments6[k][j];p->lines6[i][p->count6[i]++]=(unsigned short)k;}
        update6(c,k,1);
    }
}
static void set_point(Context *c,int point,int value) {
    Incremental *p=&c->position;
    int old=c->board[point];
    for(int k=0;k<p->count5[point];k++) update5(c,p->lines5[point][k],-1);
    for(int k=0;k<p->count6[point];k++) update6(c,p->lines6[point][k],-1);
    if(old) c->hash^=piece_key(point,old);
    if(value) c->hash^=piece_key(point,value);
    if(old==1||old==2) update_neighbors(c,point,-1);
    if(value==1||value==2) update_neighbors(c,point,1);
    p->empty_count+=(value==0)-(old==0);
    c->board[point]=(u8)value;
    for(int k=0;k<p->count5[point];k++) update5(c,p->lines5[point][k],1);
    for(int k=0;k<p->count6[point];k++) update6(c,p->lines6[point][k],1);
}
static void analyse(Context *c,Work *w) {
    Incremental *p=&c->position;
    w->empty_count=p->empty_count;
    w->win_count[0]=w->win_count[1]=0;
    w->evaluation[0]=p->evaluation[0];w->evaluation[1]=p->evaluation[1];
    int nearby=0,center=(c->rows/2)*c->cols+c->cols/2;
    for(int i=0;i<c->cells;i++) {
        for(int s=0;s<2;s++) {
            w->score[s][i]=p->score[s][i];
            w->wins[s][i]=p->wins[s][i]>0;
            w->upgrades[s][i]=p->upgrades[s][i]>0;
            w->win_count[s]+=w->wins[s][i];
        }
        w->frontier[i]=!c->board[i]&&(p->stones?p->neighbors[i]>0:!c->board[center]?i==center:1);
        nearby|=w->frontier[i];
    }
    if(!nearby) for(int i=0;i<c->cells;i++) w->frontier[i]=!c->board[i];
}
static int winning_choice(Context *c,Work *w,int side) {
    int best=-1;
    for(int i=0;i<c->cells;i++) if(w->wins[side-1][i] && (best<0||c->priors[i]>c->priors[best])) best=i;
    return best;
}
static int rank_moves(Context *c,Work *w,int side,int preferred,int root,int *complete) {
    int n=0,total=0,s=side-1,enemy=2-side;
    if(w->win_count[enemy]==1) {
        w->choices[0].move=winning_choice(c,w,3-side); w->choices[0].order=0; *complete=1; return 1;
    }
    double maximum=0;
    if(root) for(int i=0;i<c->cells;i++) if(!c->board[i]&&c->priors[i]>maximum) maximum=c->priors[i];
    for(int point=0;point<c->cells;point++) if(w->frontier[point]&&!c->board[point]) {
        total++;
        int r=2*(point/c->cols)-(c->rows-1),q=2*(point%c->cols)-(c->cols-1);
        i64 score=(i64)10*w->score[s][point]+(i64)11*w->score[enemy][point];
        if(maximum>0) score+=(int)(240*(c->priors[point]/maximum));
        score=score*16384-r*r-q*q;
        if(point==preferred) score+=((i64)1<<60);
        int at=n;
        if(at>=c->width) {at=c->width-1; if(score<=w->choices[at].order) continue;}
        else n++;
        while(at>0&&score>w->choices[at-1].order) {w->choices[at]=w->choices[at-1]; at--;}
        w->choices[at].move=point; w->choices[at].order=score;
    }
    *complete=total==w->empty_count&&total<=c->width;
    return n;
}
static int forcing_move(Context *c,Work *w,int side,int ply) {
    if(ply+1>=MAX_PLY) return -1;
    int block=w->win_count[2-side]==1?winning_choice(c,w,3-side):-1;
    for(int point=0;point<c->cells;point++) if(w->upgrades[side-1][point] && (block<0||point==block)) {
        if(!check(c)) return -1;
        c->nodes++;
        set_point(c,point,side);
        Work *child=&c->work[ply+1]; analyse(c,child);
        int proved=child->win_count[2-side]==0&&child->win_count[side-1]>=2;
        set_point(c,point,0);
        if(proved) return point;
    }
    return -1;
}
static Result search(Context *c,int side,int depth,int alpha,int beta,int ply,int last,int quiescence) {
    Result result={0,UNKNOWN};
    if(!check(c)) return result;
    c->nodes++;
    if(last>=0&&won_at(c,last,3-side)) {result.score=-MATE; result.proof=-1; return result;}
    Work *w=&c->work[ply]; analyse(c,w);
    if(w->win_count[side-1]) {result.score=MATE; result.proof=1; return result;}
    if(w->win_count[2-side]>1) {result.score=-MATE; result.proof=-1; return result;}
    if(!w->empty_count) {result.proof=0; return result;}
    u64 key=c->hash^piece_key(MAX_CELLS+1,side);
    Transposition *entry=&c->table[(unsigned int)key&(TT_SIZE-1)];
    int match=entry->generation==c->generation&&entry->key==key;
    int preferred=match?entry->move:-1,original_alpha=alpha;
    /* Cached scores remain heuristic, never proof. Do not hide exhaustive
     * small endgame proofs or reuse mate scores as certified outcomes. */
    if(match&&depth>0&&w->empty_count>12&&entry->depth>=depth&&
       entry->score>-MATE/4&&entry->score<MATE/4&&
       (entry->flag==0||(entry->flag==1&&entry->score>=beta)||(entry->flag==2&&entry->score<=alpha))) {
        c->table_hits++;result.score=entry->score;return result;
    }
    if(forcing_move(c,w,side,ply)>=0) {result.score=MATE; result.proof=1; return result;}
    if(c->aborted) return result;
    if(ply>=MAX_PLY-2 || (depth<=0&&(!w->win_count[2-side]||quiescence<=0))) {
        result.score=w->evaluation[side-1]-w->evaluation[2-side];
        if(result.score>MATE/4) result.score=MATE/4;
        if(result.score<-MATE/4) result.score=-MATE/4;
        return result;
    }
    int complete=0,count=rank_moves(c,w,side,preferred,0,&complete),known=1,bestproof=-1,explored=1,bestmove=-1;
    result.score=-MATE-1;
    for(int i=0;i<count;i++) {
        if(!check(c)) return result;
        int move=w->choices[i].move;
        set_point(c,move,side);
        Result child=search(c,3-side,depth>0?depth-1:0,-beta,-alpha,ply+1,move,quiescence-(depth<=0));
        set_point(c,move,0);
        if(c->aborted) return result;
        int score=-child.score;
        if(score>result.score) {result.score=score;bestmove=move;}
        if(child.proof==-1) {result.score=MATE; result.proof=1; return result;}
        if(child.proof==UNKNOWN) known=0;
        else if(-child.proof>bestproof) bestproof=-child.proof;
        if(score>alpha) alpha=score;
        if(alpha>=beta) {explored=0; break;}
    }
    if(complete&&known&&explored&&count>0) result.proof=bestproof;
    if(depth>0&&(!match||entry->depth<=depth)) {
        entry->key=key;entry->generation=c->generation;entry->depth=(short)depth;
        entry->score=result.score;entry->move=bestmove;
        entry->flag=result.score<=original_alpha?2:result.score>=beta?1:0;
    }
    return result;
}
EXPORT u64 native_context_size(void) {return sizeof(Context);}
EXPORT int native_transposition_hits(void *context) {return context?((Context*)context)->table_hits:0;}
EXPORT int native_select(void *context,const u8 *board,int rows,int cols,int side,const double *priors,
                         int max_depth,int width,int max_nodes,StopCallback stop,Output *out) {
    if(!context||!board||!priors||!out||rows<1||cols<1||rows>64||cols>64||(side!=1&&side!=2)||
       width<1||width>MAX_WIDTH||max_depth<1||max_depth>MAX_PLY-6||max_nodes<0) return -1;
    Context *c=(Context*)context;
    c->rows=rows;c->cols=cols;c->cells=rows*cols;c->root_side=side;c->width=width;c->max_depth=max_depth;
    c->max_nodes=max_nodes;c->nodes=0;c->aborted=0;c->stop=stop;c->n5=c->n6=0;
    c->table_hits=0;
    if(++c->generation==0) {
        for(int i=0;i<TT_SIZE;i++) c->table[i].generation=0;
        c->generation=1;
    }
    for(int i=0;i<c->cells;i++) {if(board[i]>3) return -2; c->board[i]=board[i];c->priors[i]=priors[i];}
    for(int r=0;r<rows;r++) for(int q=0;q<cols;q++) for(int d=0;d<4;d++) {
        int r5=r+4*DR[d],c5=q+4*DC[d],r6=r+5*DR[d],c6=q+5*DC[d];
        if(r5>=0&&r5<rows&&c5>=0&&c5<cols) {for(int i=0;i<5;i++) c->segments5[c->n5][i]=(r+i*DR[d])*cols+q+i*DC[d];c->n5++;}
        if(r6>=0&&r6<rows&&c6>=0&&c6<cols) {for(int i=0;i<6;i++) c->segments6[c->n6][i]=(r+i*DR[d])*cols+q+i*DC[d];c->n6++;}
    }
    out->move=-1;out->nodes=out->completed_depth=out->score=out->budget_exhausted=out->status=0;out->proof=UNKNOWN;
    for(int i=0;i<c->cells;i++) if((c->board[i]==1||c->board[i]==2)&&won_at(c,i,c->board[i])) {
        out->proof=c->board[i]==side?1:-1;out->score=out->proof*MATE;out->status=1;return 0;
    }
    initialize_position(c);
    Work *w=&c->work[0];analyse(c,w);
    if(!w->empty_count) {out->proof=0;out->status=1;return 0;}
    if(w->win_count[side-1]) {out->move=winning_choice(c,w,side);out->score=MATE;out->proof=1;out->status=2;return 0;}
    if(w->win_count[2-side]>1) {out->move=winning_choice(c,w,3-side);out->score=-MATE;out->proof=-1;out->status=3;return 0;}
    int complete=0,count=rank_moves(c,w,side,-1,1,&complete);
    if(!count) return -3;
    out->move=w->choices[0].move;
    out->score=w->evaluation[side-1]-w->evaluation[2-side];
    int forcing=forcing_move(c,w,side,0);
    if(forcing>=0) {out->move=forcing;out->proof=1;out->score=MATE;out->status=4;out->nodes=c->nodes;return 0;}
    for(int depth=1;depth<=max_depth&&!c->aborted;depth++) {
        count=rank_moves(c,w,side,out->move,1,&complete);
        int best=-MATE-1,bestmove=out->move,best_child_proof=UNKNOWN,known=1,bestproof=-1;
        for(int i=0;i<count;i++) {
            if(!check(c)) break;
            int move=w->choices[i].move;set_point(c,move,side);
            Result child=search(c,3-side,depth-1,-MATE-1,-best,1,move,4);
            set_point(c,move,0);
            if(c->aborted) break;
            int score=-child.score;
            /* A pruned child's MATE score can still be unknown. On equal
             * scores prefer it over a move whose child is proved winning.
             * This changes only the selected move, never the proof aggregate. */
            if(score>best || (score==best && best_child_proof==1 && child.proof!=1)) {
                best=score;bestmove=move;best_child_proof=child.proof;
            }
            if(child.proof==UNKNOWN) known=0;
            else if(-child.proof>bestproof) bestproof=-child.proof;
            if(child.proof==-1) {out->move=move;out->score=MATE;out->proof=1;out->completed_depth=depth;out->status=5;goto done;}
        }
        if(c->aborted) break;
        out->move=bestmove;out->score=best;out->completed_depth=depth;
        if(complete&&known) {out->proof=bestproof;break;}
    }
done:
    out->nodes=c->nodes;out->budget_exhausted=c->aborted;return 0;
}
