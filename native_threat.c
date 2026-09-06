/* Exact positive VCF/quiet threat certificates; deliberately independent of
 * the native alpha-beta proof/heuristic path. One translation unit lets the
 * existing main search share Context without changing its implementation. */
#include "native_board.c"

#define THREAT_CELLS 1024
#define THREAT_TOTAL 64
#define THREAT_NODES 2048
#define THREAT_EDGES 65536
#define THREAT_LINES 262144
#define THREAT_ITERATIONS 32
#define THREAT_REPLY_WORDS (THREAT_CELLS/32)
#define THREAT_MAP_WORDS (THREAT_CELLS*10/32)
#define THREAT_LINE_INDEX 8192
#define THREAT_LOGICAL_EDGES (THREAT_NODES*THREAT_CELLS)
#define THREAT_STORAGE_VERSION 2

typedef double (*ThreatClock)(void);
typedef struct {
    int value,move,nodes,exhausted,root_id,certificate_nodes,certificate_edges;
    int pv_length,certified_replies,completed_quiet,status,iteration_count;
} ThreatOutput;
typedef struct {
    int move,side,first_edge,edge_count;
    u8 board[THREAT_CELLS];
} ThreatNode;
typedef struct { int reply,child_id,line_offset,line_length; } ThreatEdge;
/* Only identical strict line leaves share a group, within one parent only.
 * Child graph references always keep their own physical group. */
typedef struct { int child_id,line_offset,line_length; } ThreatGroup;
typedef struct { int ordering,quiet,leaf_ms,nodes,completed; } ThreatIteration;
/* Compile-time guards for the fixed ctypes/WASM memory layout. */
typedef char ThreatOutputABI[(sizeof(ThreatOutput)==12*4)?1:-1];
typedef char ThreatNodeABI[(sizeof(ThreatNode)==4*4+THREAT_CELLS)?1:-1];
typedef char ThreatEdgeABI[(sizeof(ThreatEdge)==4*4)?1:-1];
typedef char ThreatIterationABI[(sizeof(ThreatIteration)==5*4)?1:-1];
typedef struct {
    int winner,empty_count,win_count[2];
    u8 wins[2][THREAT_CELLS];
    int forcing[2][THREAT_CELLS],score[2][THREAT_CELLS];
} ThreatFacts;
typedef struct {
    ThreatFacts facts;
    int order[THREAT_CELLS],attacks[MAX_WIDTH];
    i64 rank[THREAT_CELLS];
    ThreatEdge pending[THREAT_CELLS];
} ThreatFrame;
typedef struct {
    int value,move,root_id,length,certified_replies;
    int line[THREAT_TOTAL];
} ThreatProof;
typedef struct {
    Context base;
    ThreatFrame frames[THREAT_TOTAL+1];
    ThreatNode certificates[THREAT_NODES];
    ThreatGroup groups[THREAT_EDGES];
    int group_first[THREAT_NODES],group_count[THREAT_NODES];
    unsigned int reply_bits[THREAT_NODES][THREAT_REPLY_WORDS];
    unsigned int reply_map[THREAT_NODES][THREAT_MAP_WORDS];
    /* Offset+1 and length; range and full content checks also handle rollback. */
    unsigned int line_index[THREAT_LINE_INDEX];
    ThreatEdge edge_scratch;
    int lines[THREAT_LINES],pv[THREAT_TOTAL];
    ThreatIteration iterations[THREAT_ITERATIONS];
    u8 initial_board[THREAT_CELLS];
    int attacker,quiet_limit,total,width,nodes,max_nodes,phase_nodes;
    int node_capacity,edge_capacity,line_capacity;
    int used_nodes,used_edges,used_groups,used_lines,pv_length,iteration_count;
    int ordering,leaf_ms,leaf_nodes,stop_kind,clock_error,completed_quiet;
    double started,deadline,phase_deadline,last_now;
    ThreatClock now;
} ThreatContext;
/* Keep the current 16 MiB initial memory, including the 256 KiB stack and
 * adapter/global headroom. No change to the 32 MiB memory ceiling. */
typedef char ThreatInitialMemoryBound[(sizeof(ThreatContext)+262144+131072<16777216)?1:-1];

EXPORT u64 native_threat_context_size(void) { return sizeof(ThreatContext); }
EXPORT int native_threat_storage_version(void) { return THREAT_STORAGE_VERSION; }
EXPORT int native_threat_group_count(void *context) {
    return context?((ThreatContext*)context)->used_groups:0;
}
EXPORT int native_threat_group_capacity(void *context) {
    return context?((ThreatContext*)context)->edge_capacity:0;
}
EXPORT void *native_threat_base_context(void *context) {
    return context ? &((ThreatContext*)context)->base : (void*)0;
}
EXPORT ThreatNode *native_threat_node(void *context,int id) {
    ThreatContext *t=(ThreatContext*)context;
    return t && id>=0 && id<t->used_nodes ? &t->certificates[id] : (ThreatNode*)0;
}
EXPORT ThreatEdge *native_threat_edge(void *context,int index) {
    ThreatContext *t=(ThreatContext*)context;
    if(!t || index<0 || index>=t->used_edges) return (ThreatEdge*)0;
    /* Nodes append in logical edge-range order. Locate the parent in O(log N)
     * and its reply by rank in at most 32 bitmap words. */
    int low=0,high=t->used_nodes,id=-1;
    while(low<high) {
        int middle=low+(high-low)/2;
        ThreatNode *node=&t->certificates[middle];
        if(index<node->first_edge) high=middle;
        else if(index>=node->first_edge+node->edge_count) low=middle+1;
        else {id=middle;break;}
    }
    if(id<0) return (ThreatEdge*)0;
    int ordinal=index-t->certificates[id].first_edge,reply=-1;
    for(int w=0;w<THREAT_REPLY_WORDS;w++) {
        unsigned int bits=t->reply_bits[id][w],count=bits;
        count=count-((count>>1)&0x55555555u);
        count=(count&0x33333333u)+((count>>2)&0x33333333u);
        count=(count+(count>>4))&0x0f0f0f0fu;
        count=(count*0x01010101u)>>24;
        if(ordinal>=(int)count) {ordinal-=(int)count;continue;}
        while(ordinal-->0) bits&=bits-1;
        int bit=0;
        while(!(bits&1u)) {bits>>=1;bit++;}
        reply=w*32+bit;break;
    }
    if(reply<0 || reply>=t->base.cells) return (ThreatEdge*)0;
    int bit=reply*10,word=bit/32,shift=bit%32;
    unsigned int relative=t->reply_map[id][word]>>shift;
    if(shift>22) relative|=t->reply_map[id][word+1]<<(32-shift);
    relative&=1023u;
    if(relative>=(unsigned int)t->group_count[id]) return (ThreatEdge*)0;
    int at=t->group_first[id]+(int)relative;
    if(at<0 || at>=t->used_groups) return (ThreatEdge*)0;
    ThreatGroup *group=&t->groups[at];
    /* Copy all four fields before the next edge getter: one scratch record. */
    t->edge_scratch.reply=reply;t->edge_scratch.child_id=group->child_id;
    t->edge_scratch.line_offset=group->line_offset;t->edge_scratch.line_length=group->line_length;
    return &t->edge_scratch;
}
EXPORT int *native_threat_lines(void *context) {
    return context ? ((ThreatContext*)context)->lines : (int*)0;
}
EXPORT int native_threat_line_count(void *context) {
    return context ? ((ThreatContext*)context)->used_lines : 0;
}
EXPORT int *native_threat_pv(void *context) {
    return context ? ((ThreatContext*)context)->pv : (int*)0;
}
EXPORT ThreatIteration *native_threat_iteration(void *context,int index) {
    ThreatContext *t=(ThreatContext*)context;
    return t && index>=0 && index<t->iteration_count ? &t->iterations[index] : (ThreatIteration*)0;
}
EXPORT u8 *native_threat_board(void *context) {
    return context ? ((ThreatContext*)context)->base.board : (u8*)0;
}
EXPORT int native_threat_solve(void *context,const u8 *board,int rows,int cols,int side,
    double millis,int max_nodes,int quiet,int total,int width,int node_capacity,
    int edge_capacity,int line_capacity,ThreatClock now,ThreatOutput *out);
EXPORT int native_threat_solve_range(void *context,const u8 *board,int rows,int cols,int side,
    double millis,int max_nodes,int quiet,int total,int width,int node_capacity,
    int edge_capacity,int line_capacity,ThreatClock now,ThreatOutput *out,int min_quiet);

/* Checked clock and one shared node/deadline budget. Local VCF exhaustion is
 * deliberately separate from phase/global exhaustion, so another candidate may
 * still be attempted within the original allocation. */
static int threat_finite(double value) {
    return value==value && value<=1.7976931348623157e308 && value>=-1.7976931348623157e308;
}
static double threat_time(ThreatContext *t) {
    double value=t->now();
    if(!threat_finite(value) || value<t->last_now) {
        t->clock_error=1;t->stop_kind=2;return t->last_now;
    }
    t->last_now=value;return value;
}
static int threat_check(ThreatContext *t,int force_clock) {
    if(t->stop_kind) return 0;
    if(t->nodes>=t->max_nodes || t->nodes>=t->phase_nodes) {
        t->stop_kind=2;return 0;
    }
    if(force_clock || (t->nodes&31)==0) {
        double now=threat_time(t);
        if(t->clock_error || now>=t->deadline || now>=t->phase_deadline) {
            t->stop_kind=2;return 0;
        }
    }
    return 1;
}
static int threat_take(ThreatContext *t) {
    if(!threat_check(t,0)) return 0;
    t->nodes++;return 1;
}
static ThreatProof threat_unknown(void) {
    ThreatProof p={0};p.value=UNKNOWN;p.move=p.root_id=-1;return p;
}
static ThreatProof threat_fact_proof(int value,int move) {
    ThreatProof p=threat_unknown();p.value=value;p.move=move;
    if(move>=0) {p.length=1;p.line[0]=move;}
    return p;
}
static int threat_first(const u8 *mask,int cells,int except) {
    for(int p=0;p<cells;p++) if(mask[p] && p!=except) return p;
    return -1;
}

/* Exactly the browser's five-cell facts/ordering. No six-cell bonuses,
 * frontier restriction, neural priors or alpha-beta scores enter this prover. */
static void threat_facts(ThreatContext *t,ThreatFacts *f,int need_score) {
    Context *c=&t->base;
    f->winner=f->empty_count=f->win_count[0]=f->win_count[1]=0;
    for(int p=0;p<c->cells;p++) {
        f->empty_count+=c->board[p]==0;
        for(int s=0;s<2;s++) {
            f->wins[s][p]=0;f->forcing[s][p]=0;
            if(need_score) f->score[s][p]=0;
        }
    }
    for(int k=0;k<c->n5;k++) {
        int *line=c->segments5[k],counts[4]={0,0,0,0},empty[5],used=0;
        for(int i=0;i<5;i++) {
            int p=line[i],value=c->board[p];counts[value]++;
            if(!value) empty[used++]=p;
        }
        if(counts[1]==5) f->winner=1;
        if(counts[2]==5) f->winner=2;
        if(counts[3]) continue;
        for(int s=0;s<2;s++) if(!counts[2-s]) {
            int own=counts[s+1];
            if(own==4 && used==1 && !f->wins[s][empty[0]]) {
                f->wins[s][empty[0]]=1;f->win_count[s]++;
            }
            if(own==3 && used==2) for(int i=0;i<used;i++) f->forcing[s][empty[i]]++;
            if(need_score && own<5) {
                int delta=WEIGHTS[own+1]-WEIGHTS[own];
                for(int i=0;i<used;i++) f->score[s][empty[i]]+=delta;
            }
        }
    }
}
static int threat_worse(ThreatFrame *frame,int a,int b) {
    return frame->rank[a]<frame->rank[b] ||
        (frame->rank[a]==frame->rank[b] && a>b);
}
static void threat_sift(ThreatFrame *frame,int start,int count) {
    int root=start;
    while(root*2+1<count) {
        int child=root*2+1;
        if(child+1<count && threat_worse(frame,frame->order[child+1],frame->order[child])) child++;
        if(!threat_worse(frame,frame->order[child],frame->order[root])) break;
        int tmp=frame->order[root];frame->order[root]=frame->order[child];frame->order[child]=tmp;
        root=child;
    }
}
static int threat_order(ThreatContext *t,ThreatFrame *frame,int side,int forcing_only) {
    Context *c=&t->base;ThreatFacts *f=&frame->facts;
    int count=0,s=side-1,enemy=2-side;
    for(int point=0;point<c->cells;point++) if(!c->board[point] &&
            (!forcing_only || f->forcing[s][point])) {
        frame->order[count++]=point;
        if(forcing_only) frame->rank[point]=f->forcing[s][point];
        else {
            int r=2*(point/c->cols)-(c->rows-1),q=2*(point%c->cols)-(c->cols-1);
            frame->rank[point]=((i64)10*f->score[s][point]+(i64)11*f->score[enemy][point])*16384-r*r-q*q;
        }
    }
    for(int start=count/2-1;start>=0;start--) threat_sift(frame,start,count);
    for(int end=count-1;end>0;end--) {
        int tmp=frame->order[0];frame->order[0]=frame->order[end];frame->order[end]=tmp;
        threat_sift(frame,0,end);
    }
    return count;
}
typedef struct { int limit,exhausted;double deadline; } ThreatLeafBudget;
static int threat_leaf_check(ThreatContext *t,ThreatLeafBudget *leaf,int force_clock) {
    if(!threat_check(t,force_clock)) return 0;
    if(t->nodes>=leaf->limit || t->last_now>=leaf->deadline) {
        leaf->exhausted=1;return 0;
    }
    return 1;
}

/* Internal negamax VCF. A negative continuation is propagated only through a
 * unique mandatory defense or the exact two-winning-cells fact. Enumerating a
 * non-forcing defender choice needs an AND certificate, handled by quiet_visit,
 * and must not be represented by one allegedly forced principal variation. */
static ThreatProof threat_vcf_visit(ThreatContext *t,int actor,int depth,int ply,
                                   int line_cap,ThreatLeafBudget *leaf) {
    ThreatProof unknown=threat_unknown();
    if(!threat_leaf_check(t,leaf,0)) return unknown;
    t->nodes++;
    if(ply>THREAT_TOTAL) return unknown;
    ThreatFrame *frame=&t->frames[ply];
    ThreatFacts *f=&frame->facts;
    threat_facts(t,f,0);
    if(f->winner) return threat_fact_proof(f->winner==actor?1:-1,-1);
    if(f->win_count[actor-1]) {
        if(line_cap<1) return unknown;
        return threat_fact_proof(1,threat_first(f->wins[actor-1],t->base.cells,-1));
    }
    if(f->win_count[2-actor]>1) {
        if(line_cap<2) return unknown;
        ThreatProof proof=threat_fact_proof(-1,threat_first(f->wins[2-actor],t->base.cells,-1));
        proof.line[1]=threat_first(f->wins[2-actor],t->base.cells,proof.move);
        proof.length=2;return proof;
    }
    if(!f->empty_count) return threat_fact_proof(0,-1);
    if(depth<=0 || ply>=THREAT_TOTAL || line_cap<=0) return unknown;
    int mandatory=f->win_count[2-actor]==1,count;
    if(mandatory) {
        frame->order[0]=threat_first(f->wins[2-actor],t->base.cells,-1);count=1;
    } else count=threat_order(t,frame,actor,1);
    ThreatProof best=unknown;
    int known=1;
    for(int i=0;i<count;i++) {
        if(!threat_leaf_check(t,leaf,0)) return unknown;
        int move=frame->order[i];t->base.board[move]=(u8)actor;
        ThreatProof child=threat_vcf_visit(t,3-actor,depth-1,ply+1,line_cap-1,leaf);
        t->base.board[move]=0;
        if(t->stop_kind || leaf->exhausted) return unknown;
        if(child.value==UNKNOWN || child.length+1>line_cap) {known=0;continue;}
        ThreatProof proof=threat_fact_proof(-child.value,move);
        proof.length=child.length+1;
        for(int k=0;k<child.length;k++) proof.line[k+1]=child.line[k];
        if(proof.value==1) return proof;
        if(best.value==UNKNOWN || proof.value>best.value ||
                (proof.value==best.value && proof.length>best.length)) best=proof;
    }
    return mandatory && known ? best : unknown;
}
static ThreatProof threat_vcf_leaf(ThreatContext *t,int ply,int line_cap) {
    ThreatProof unknown=threat_unknown();
    if(!threat_check(t,1)) return unknown;
    double now=t->last_now;
    ThreatLeafBudget leaf;
    leaf.deadline=now+t->leaf_ms;
    if(leaf.deadline>t->phase_deadline) leaf.deadline=t->phase_deadline;
    if(leaf.deadline>t->deadline) leaf.deadline=t->deadline;
    int remaining=t->phase_nodes-t->nodes;
    if(remaining>t->max_nodes-t->nodes) remaining=t->max_nodes-t->nodes;
    if(remaining>t->leaf_nodes) remaining=t->leaf_nodes;
    leaf.limit=t->nodes+remaining;leaf.exhausted=0;
    int depth=line_cap>2?line_cap-2:0;
    if(depth>32) depth=32;
    return threat_vcf_visit(t,t->attacker,depth,ply,line_cap,&leaf);
}
static int threat_same_line(ThreatContext *t,int left,int right,int length) {
    if(left<0 || right<0 || length<0 || left>t->used_lines || right>t->used_lines ||
            length>t->used_lines-left || length>t->used_lines-right) return 0;
    for(int i=0;i<length;i++) if(t->lines[left+i]!=t->lines[right+i]) return 0;
    return 1;
}
static int threat_line_store(ThreatContext *t,const ThreatProof *proof) {
    if(proof->length<0 || proof->length>THREAT_TOTAL) return -1;
    if(!proof->length) return t->used_lines;
    unsigned int hash=2166136261u;
    for(int i=0;i<proof->length;i++) hash=(hash^(unsigned int)(proof->line[i]+1))*16777619u;
    hash=(hash^(unsigned int)proof->length)*16777619u;
    int available=-1;
    for(int probe=0;probe<THREAT_LINE_INDEX;probe++) {
        int slot=(int)((hash+(unsigned int)probe)&(THREAT_LINE_INDEX-1));
        unsigned int entry=t->line_index[slot];
        if(!entry) {if(available<0) available=slot;break;}
        int offset=(int)(entry&524287u)-1,length=(int)(entry>>19);
        int live=offset>=0 && offset<=t->used_lines && length>0 &&
                 length<=THREAT_TOTAL && length<=t->used_lines-offset;
        if(!live) {if(available<0) available=slot;}
        else if(length==proof->length) {
            int same=1;
            for(int i=0;i<length;i++) if(t->lines[offset+i]!=proof->line[i]) {same=0;break;}
            if(same) return offset;
        }
        /* A full index is only an interning miss, never evidence. Keep even a
         * collision-heavy storage lookup inside the shared deadline. */
        if((probe&255)==255 && !threat_check(t,1)) return -1;
    }
    if(proof->length>t->line_capacity-t->used_lines) {t->stop_kind=3;return -1;}
    int offset=t->used_lines;
    for(int i=0;i<proof->length;i++) t->lines[t->used_lines++]=proof->line[i];
    if(available>=0) t->line_index[available]=((unsigned int)proof->length<<19)|(unsigned int)(offset+1);
    return offset;
}
static void threat_rollback(ThreatContext *t,int nodes,int edges,int lines,int groups) {
    t->used_nodes=nodes;t->used_edges=edges;t->used_lines=lines;t->used_groups=groups;
    /* Index entries are not proof records. A later lookup validates both the
     * live prefix and every move, even if rolled-back bytes were overwritten.
     * Keeping occupied index slots preserves collision probe chains. */
}
static int threat_node_store(ThreatContext *t,int move,const ThreatEdge *pending,int count) {
    if(count<=0 || count>t->base.cells) return -1;
    if(t->used_nodes>=t->node_capacity || count>THREAT_LOGICAL_EDGES-t->used_edges) {
        t->stop_kind=3;return -1;
    }
    int id=t->used_nodes,first=t->used_groups;
    for(int w=0;w<THREAT_REPLY_WORDS;w++) t->reply_bits[id][w]=0;
    for(int w=0;w<THREAT_MAP_WORDS;w++) t->reply_map[id][w]=0;
    for(int i=0;i<count;i++) {
        if((i&31)==0 && !threat_check(t,1)) return -1;
        const ThreatEdge *edge=&pending[i];
        int reply=edge->reply;
        if(reply<0 || reply>=t->base.cells || t->base.board[reply] ||
                edge->child_id< -1 || edge->child_id>=id ||
                !threat_same_line(t,edge->line_offset,edge->line_offset,edge->line_length)) return -1;
        unsigned int bit=1u<<(reply%32);
        if(t->reply_bits[id][reply/32]&bit) return -1;
        int group=-1;
        if(edge->child_id==-1) for(int g=first;g<t->used_groups;g++) {
            ThreatGroup *existing=&t->groups[g];
            if(existing->child_id==-1 && existing->line_length==edge->line_length &&
                    threat_same_line(t,existing->line_offset,edge->line_offset,edge->line_length)) {
                group=g;break;
            }
            if(((g-first)&255)==255 && !threat_check(t,1)) return -1;
        }
        if(group<0) {
            if(t->used_groups>=t->edge_capacity) {t->stop_kind=3;return -1;}
            group=t->used_groups++;
            t->groups[group].child_id=edge->child_id;
            t->groups[group].line_offset=edge->line_offset;
            t->groups[group].line_length=edge->line_length;
        }
        unsigned int relative=(unsigned int)(group-first);
        if(relative>1023u) return -1;
        int map_bit=reply*10,word=map_bit/32,shift=map_bit%32;
        t->reply_map[id][word]=(t->reply_map[id][word]&~(1023u<<shift))|(relative<<shift);
        if(shift>22) {
            unsigned int mask=(1u<<(shift-22))-1u;
            t->reply_map[id][word+1]=(t->reply_map[id][word+1]&~mask)|(relative>>(32-shift));
        }
        t->reply_bits[id][reply/32]|=bit;
    }
    /* Explicitly verify the stored bitmap equals every real legal reply.
     * Groups only compress proofs already obtained by the full defender loop. */
    for(int p=0;p<t->base.cells;p++)
        if((int)((t->reply_bits[id][p/32]>>(p%32))&1u)!=(t->base.board[p]==0)) return -1;
    ThreatNode *node=&t->certificates[id];
    node->move=move;node->side=t->attacker;
    node->first_edge=t->used_edges;node->edge_count=count;
    for(int p=0;p<THREAT_CELLS;p++) node->board[p]=p<t->base.cells?t->base.board[p]:0;
    t->group_first[id]=first;t->group_count[id]=t->used_groups-first;
    /* Publish only after the complete bitmap and every group are committed.
     * Children were published first, so all child IDs stay below this ID. */
    t->used_edges+=count;t->used_nodes++;
    return id;
}
static ThreatProof threat_quiet_visit(ThreatContext *t,int left,int ply) {
    ThreatProof unknown=threat_unknown();
    if(!threat_take(t)) return unknown;
    if(ply>THREAT_TOTAL) return unknown;
    ThreatFrame *frame=&t->frames[ply];
    ThreatFacts *f=&frame->facts;
    threat_facts(t,f,1);
    if(f->winner) return f->winner==t->attacker?threat_fact_proof(1,-1):unknown;
    if(ply>=t->total) return unknown;
    if(f->win_count[t->attacker-1])
        return threat_fact_proof(1,threat_first(f->wins[t->attacker-1],t->base.cells,-1));
    int mandatory=f->win_count[2-t->attacker]==1;
    if(f->win_count[2-t->attacker]>1 || !f->empty_count ||
            ply+2>=t->total) return unknown;
    if(!mandatory && left<=0) {
        /* A forcing attack spends no quiet ply. Keep this strict line proof
           inside the current iteration and its shared node/time budget. */
        ThreatProof forcing=threat_vcf_leaf(t,ply,t->total-ply);
        return forcing.value==1?forcing:unknown;
    }
    int count=0;
    if(mandatory) frame->attacks[count++]=threat_first(f->wins[2-t->attacker],t->base.cells,-1);
    else {
        int ordered=threat_order(t,frame,t->attacker,0);
        if(t->ordering==0) {
            for(int pass=0;pass<2;pass++) for(int i=0;i<ordered && count<t->width;i++) {
                int move=frame->order[i],forcing=f->forcing[t->attacker-1][move]>0;
                if(forcing==pass) frame->attacks[count++]=move;
            }
        } else for(int i=0;i<ordered && count<t->width;i++) frame->attacks[count++]=frame->order[i];
    }
    for(int a=0;a<count;a++) {
        if(!threat_take(t)) return unknown;
        int move=frame->attacks[a];
        int before_nodes=t->used_nodes,before_edges=t->used_edges,before_lines=t->used_lines,before_groups=t->used_groups;
        t->base.board[move]=(u8)t->attacker;
        ThreatFrame *defense=&t->frames[ply+1];
        ThreatFacts *cf=&defense->facts;
        threat_facts(t,cf,1);
        int complete=1,used=0,reply_count=0;
        ThreatProof longest=threat_unknown(),answer=unknown;
        if(cf->win_count[2-t->attacker]) complete=0;
        int next_quiet=left-((mandatory || cf->win_count[t->attacker-1])?0:1);
        if(complete) {
            reply_count=threat_order(t,defense,3-t->attacker,0);
            if(!reply_count) complete=0;
        }
        for(int r=0;complete && r<reply_count;r++) {
            if(!threat_take(t)) {complete=0;break;}
            int reply=defense->order[r],cap=t->total-ply-2;
            t->base.board[reply]=(u8)(3-t->attacker);
            ThreatProof proof=unknown;
            int immediate=threat_first(cf->wins[t->attacker-1],t->base.cells,reply);
            /* Before the reply the opponent had no immediate win. A reply
             * outside our single-empty five cannot interrupt its completion. */
            if(cap>=1 && immediate>=0 && !t->base.board[immediate]) {
                if(threat_take(t)) proof=threat_fact_proof(1,immediate);
            } else {
                proof=threat_vcf_leaf(t,ply+2,cap);
            }
            if(proof.value==UNKNOWN && !t->stop_kind) {
                int recurse=next_quiet>0;
                if(next_quiet==0) {
                    ThreatFacts *next=&t->frames[ply+2].facts;
                    threat_facts(t,next,0);
                    recurse=next->win_count[2-t->attacker]==1;
                }
                if(recurse) proof=threat_quiet_visit(t,next_quiet,ply+2);
            }
            if(t->stop_kind || proof.value!=1 || proof.length>cap) complete=0;
            if(complete) {
                ThreatEdge edge;
                edge.reply=reply;edge.child_id=proof.root_id;
                /* Every edge carries its representative continuation as well
                 * as an optional complete child certificate. */
                edge.line_offset=threat_line_store(t,&proof);
                edge.line_length=proof.length;
                if(t->stop_kind) complete=0;
                if(complete) {
                    defense->pending[used++]=edge;
                    if(proof.length+1>longest.length) {
                        longest.length=proof.length+1;longest.line[0]=reply;
                        for(int k=0;k<proof.length;k++) longest.line[k+1]=proof.line[k];
                    }
                }
            }
            t->base.board[reply]=0;
        }
        if(complete && used==reply_count) {
            int id=threat_node_store(t,move,defense->pending,used);
            if(id<0) complete=0;
            else {
                answer=threat_fact_proof(1,move);
                answer.root_id=id;answer.certified_replies=used;answer.length=longest.length+1;
                for(int i=0;i<longest.length;i++) answer.line[i+1]=longest.line[i];
            }
        }
        t->base.board[move]=0;
        if(complete && answer.value==1) return answer;
        threat_rollback(t,before_nodes,before_edges,before_lines,before_groups);
        if(t->stop_kind) return unknown;
    }
    return unknown;
}

/* Host-visible execution contract:
 * 0 returns either a complete positive certificate or an explicit unknown.
 * -1 invalid pointers/ranges/time allowance, -2 invalid cell encoding,
 * -3 a nonfinite or backwards host clock. Invalid clocks discard every proof.
 * Status: 0 completed bounded unknown, 1 positive proof, 2 budget, 3 arena
 * capacity, 4 terminal input. Positive results always have exhausted=0.
 *
 * VCF leaves intentionally certify a narrower class than JS's generic
 * complete-legal-set fallback: negative VCF results require an immediate
 * double winning-cell fact or a unique mandatory continuation. This makes
 * each line-only edge independently checkable as a strict forcing line.
 */
/* Storage v2: edge_capacity bounds physical groups, not logical replies.
 * The original solve starts at quiet 1 (or quiet 0 for the zero-quiet case). */
EXPORT int native_threat_solve(void *context,const u8 *board,int rows,int cols,int side,
    double millis,int max_nodes,int quiet,int total,int width,int node_capacity,
    int edge_capacity,int line_capacity,ThreatClock now,ThreatOutput *out) {
    return native_threat_solve_range(context,board,rows,cols,side,millis,max_nodes,
        quiet,total,width,node_capacity,edge_capacity,line_capacity,now,out,quiet==0?0:1);
}
EXPORT int native_threat_solve_range(void *context,const u8 *board,int rows,int cols,int side,
    double millis,int max_nodes,int quiet,int total,int width,int node_capacity,
    int edge_capacity,int line_capacity,ThreatClock now,ThreatOutput *out,int min_quiet) {
    ThreatOutput empty={UNKNOWN,-1,0,0,-1,0,0,0,0,0,0,0};
    if(out) *out=empty;
    if(!context || !board || !out || !now || rows<5 || rows>32 || cols<5 || cols>32 ||
            (side!=1 && side!=2) || !threat_finite(millis) || millis<0 ||
            max_nodes<0 || quiet<0 || quiet>4 || total<1 || total>THREAT_TOTAL ||
            (quiet==0?min_quiet!=0:(min_quiet<1 || min_quiet>quiet)) ||
            width<1 || width>MAX_WIDTH || node_capacity<0 || node_capacity>THREAT_NODES ||
            edge_capacity<0 || edge_capacity>THREAT_EDGES ||
            line_capacity<0 || line_capacity>THREAT_LINES) return -1;

    ThreatContext *t=(ThreatContext*)context;
    Context *c=&t->base;
    /* Reset counts before exposing this invocation. Arena contents outside
     * these counts are stale and cannot be retrieved through checked getters. */
    t->used_nodes=t->used_edges=t->used_groups=t->used_lines=t->pv_length=t->iteration_count=0;
    t->nodes=t->stop_kind=t->clock_error=t->completed_quiet=0;
    t->attacker=side;t->quiet_limit=quiet;t->total=total;t->width=width;
    t->max_nodes=max_nodes;t->phase_nodes=max_nodes;
    t->node_capacity=node_capacity;t->edge_capacity=edge_capacity;t->line_capacity=line_capacity;
    t->ordering=0;t->leaf_ms=5;t->leaf_nodes=500;t->now=now;
    t->started=now();
    if(!threat_finite(t->started)) {t->clock_error=1;return -3;}
    t->last_now=t->started;t->deadline=t->started+millis;t->phase_deadline=t->deadline;
    if(!threat_finite(t->deadline)) return -1;
    for(int i=0;i<THREAT_LINE_INDEX;i++) t->line_index[i]=0;
    int cells=rows*cols;
    for(int p=0;p<cells;p++) if(board[p]>3) return -2;
    /* The clock above precedes board/geometry initialization. All setup work
     * therefore belongs to this same deadline, including context reuse. */
    c->rows=rows;c->cols=cols;c->cells=cells;c->root_side=side;c->n5=c->n6=0;
    for(int p=0;p<cells;p++) t->initial_board[p]=board[p];
    for(int p=0;p<cells;p++) c->board[p]=t->initial_board[p];
    for(int r=0;r<rows;r++) for(int col=0;col<cols;col++) for(int d=0;d<4;d++) {
        int end_r=r+4*DR[d],end_c=col+4*DC[d];
        if(end_r>=0 && end_r<rows && end_c>=0 && end_c<cols) {
            for(int k=0;k<5;k++) c->segments5[c->n5][k]=(r+k*DR[d])*cols+col+k*DC[d];
            c->n5++;
        }
    }

    ThreatProof proof=threat_unknown();
    int terminal=0;
    /* A zero allocation performs no visit, even for an immediate/terminal
     * board. Positive-budget root facts may finish without an ordering pass. */
    if(!threat_check(t,1)) goto finish;
    threat_facts(t,&t->frames[0].facts,0);
    if(!threat_check(t,1)) goto finish;
    ThreatFacts *root=&t->frames[0].facts;
    terminal=root->winner || !root->empty_count;
    if(terminal || root->win_count[side-1]) {
        if(!threat_take(t)) goto finish;
        if(terminal) {
            if(root->winner==side) proof=threat_fact_proof(1,-1);
        } else proof=threat_fact_proof(1,threat_first(root->wins[side-1],cells,-1));
        goto finish;
    }

    /* Same staged attacker scheduling as the JS prover. Neither switching the
     * ordering nor increasing a VCF leaf allowance resets nodes or deadline. */
    for(int order=0;order<2;order++) {
        t->ordering=order;
        t->phase_deadline=order==0?t->started+millis*0.45:t->deadline;
        if(t->phase_deadline>t->deadline) t->phase_deadline=t->deadline;
        if(order==0) {
            t->phase_nodes=(int)(((i64)max_nodes*7)/10);
            if(t->phase_nodes>3500) t->phase_nodes=3500;
            if(t->phase_nodes>max_nodes) t->phase_nodes=max_nodes;
        } else t->phase_nodes=max_nodes;
        t->stop_kind=0;
        for(int tier=0;tier<2;tier++) {
            t->leaf_ms=tier==0?5:50;t->leaf_nodes=tier==0?500:5000;
            for(int q=min_quiet;q<=quiet;q++) {
                int before=t->nodes;
                proof=threat_quiet_visit(t,q,0);
                /* A final forced clock check prevents an unpolled overrun
                 * from publishing a positive certificate for this phase. */
                if(proof.value==1) {
                    double done=threat_time(t);
                    if(t->clock_error || done>=t->deadline || done>=t->phase_deadline) {
                        proof=threat_unknown();t->stop_kind=2;
                    }
                }
                ThreatIteration *iteration=&t->iterations[t->iteration_count++];
                iteration->ordering=order;iteration->quiet=q;iteration->leaf_ms=t->leaf_ms;
                iteration->nodes=t->nodes-before;iteration->completed=t->stop_kind==0;
                if(iteration->completed) t->completed_quiet=q;
                if(proof.value==1) goto finish;
                /* Failed attempts keep no orphan subtrees or line ranges. */
                threat_rollback(t,0,0,0,0);
                if(t->stop_kind) break;
            }
            if(t->stop_kind) break;
        }
        if(t->stop_kind==3 || t->clock_error) break;
        double current=threat_time(t);
        if(t->clock_error || current>=t->deadline || t->nodes>=max_nodes) {
            t->stop_kind=2;break;
        }
        /* Only the first phase may have reached its smaller private ceiling.
         * A bounded unknown or a phase stop still leaves the natural pass. */
    }

finish:
    /* Explicit restoration is in addition to each recursive move's undo. The
     * shared base Context is safe for subsequent alpha-beta or threat calls. */
    for(int p=0;p<cells;p++) c->board[p]=t->initial_board[p];
    double ended=threat_time(t);
    if(t->clock_error || ended>=t->deadline) {
        proof=threat_unknown();
        if(t->stop_kind!=3) t->stop_kind=2;
        terminal=0;
    }
    if(proof.value!=1) threat_rollback(t,0,0,0,0);
    if(proof.value==1) {
        t->pv_length=proof.length;
        for(int k=0;k<proof.length;k++) t->pv[k]=proof.line[k];
        out->value=1;out->move=proof.move;out->root_id=proof.root_id;
        out->pv_length=proof.length;out->certified_replies=proof.certified_replies;
        out->status=terminal?4:1;out->exhausted=0;
    } else {
        out->status=terminal?4:t->stop_kind==3?3:
            (t->stop_kind==2 || t->nodes>=max_nodes)?2:0;
        out->exhausted=out->status==2 || out->status==3;
    }
    out->nodes=t->nodes;
    out->certificate_nodes=t->used_nodes;out->certificate_edges=t->used_edges;
    out->completed_quiet=t->completed_quiet;out->iteration_count=t->iteration_count;
    return t->clock_error?-3:0;
}
