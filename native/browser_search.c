/* Browser adapter. The production native core stays shared with Python. */
#include "../native_threat.c"
__attribute__((import_module("env"), import_name("now"))) extern double web_now(void);
static ThreatContext web_threat_context;
static ThreatOutput web_threat_output;
static u8 web_board[MAX_CELLS];
static double web_priors[MAX_CELLS];
static Output web_output;
static double web_deadline;
static int web_stop(void) { return web_now() >= web_deadline; }
EXPORT u8 *browser_board(void) { return web_board; }
EXPORT double *browser_priors(void) { return web_priors; }
EXPORT Output *browser_output(void) { return &web_output; }
EXPORT int browser_select(int rows,int cols,int side,double millis,int nodes,int depth,int width) {
    web_deadline=web_now()+millis;
    return native_select(native_threat_base_context(&web_threat_context),web_board,rows,cols,side,web_priors,depth,width,nodes,web_stop,&web_output);
}

/* Both entry points are serialized by the Worker and share the base scratch
 * context. Full proof records remain accessible for independent diagnostics. */
EXPORT void *browser_threat_context(void) { return &web_threat_context; }
EXPORT ThreatOutput *browser_threat_output(void) { return &web_threat_output; }
EXPORT int *browser_threat_pv(void) { return native_threat_pv(&web_threat_context); }
EXPORT int browser_threat_solve(int rows,int cols,int side,double millis,int nodes,
                               int quiet,int total,int width) {
    return native_threat_solve(&web_threat_context,web_board,rows,cols,side,millis,
        nodes,quiet,total,width,2048,65536,262144,web_now,&web_threat_output);
}

/* A fresh bounded invocation restricted to [min_quiet, quiet]. It does not
 * resume an earlier DFS and retains the same shared two-order phase policy. */
EXPORT int browser_threat_solve_range(int rows,int cols,int side,double millis,int nodes,
                                     int quiet,int total,int width,int min_quiet) {
    return native_threat_solve_range(&web_threat_context,web_board,rows,cols,side,millis,
        nodes,quiet,total,width,2048,65536,262144,web_now,&web_threat_output,min_quiet);
}
