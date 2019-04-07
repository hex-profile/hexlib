#include "profilerImpl.h"

#include "errorLog/errorLog.h"
#include "errorLog/debugBreak.h"
#include "dataAlloc/arrayObjMem.inl"
#include "storage/rememberCleanup.h"

//================================================================
//
// ProfilerScopeEx
//
//================================================================

struct ProfilerScopeEx
{
    Profiler* profiler; // The structure SHOULD start with it (used in main profiler header)
    ProfilerNode* prevCurrentScope;
    ProfilerMoment startMoment;
};

COMPILE_ASSERT(sizeof(ProfilerScopeEx) <= 24);

//================================================================
//
// NodePtr
//
//================================================================

using NodePtr = ProfilerNode*;

//================================================================
//
// ProfilerImpl::realloc
//
//================================================================

bool ProfilerImpl::realloc(Space capacity, stdPars(AllocKit))
{
    stdBegin;

    dealloc();

    //
    // Normal records pool
    //

    REQUIRE(capacity >= 1);
    REQUIRE(nodePool.realloc(capacity, cpuBaseByteAlignment, kit.malloc, stdPass));

    ARRAY_EXPOSE(nodePool);
    REQUIRE(nodePoolSize >= 1);
    currentScope = unsafePtr(nodePoolPtr, 1);
    currentScope->init(CT("<all>"));

    nodeCount = 1;

    ////

    stdEnd;
}

//================================================================
//
// ProfilerImpl::dealloc
//
//================================================================

void ProfilerImpl::dealloc()
{
    nodeCount = 0;
    currentScope = 0;

    nodePool.dealloc();
    nodeRefOwner.disconnectAll();
}

//================================================================
//
// ProfilerImpl::resetMemory
//
//================================================================

void ProfilerImpl::resetMemory()
{
    nodeCount = 0;
    nodeRefOwner.disconnectAll();

    ARRAY_EXPOSE(nodePool);
    requirev(nodePoolSize >= 1);
    currentScope = unsafePtr(nodePoolPtr, 1);
    currentScope->init(CT("<all>"));
    nodeCount = 1;
}

//================================================================
//
// ProfilerImpl::checkResetScope
//
//================================================================

bool ProfilerImpl::checkResetScope()
{
    ARRAY_EXPOSE(nodePool);
    require(nodePoolSize >= 1);

    ////

    bool ok = (currentScope == unsafePtr(nodePoolPtr, 1));
    currentScope = unsafePtr(nodePoolPtr, 1);

    return ok;
}

//================================================================
//
// ProfilerImpl::enter
//
//================================================================

inline void ProfilerImpl::enter(ProfilerScopeEx& scope, TraceLocation location, uint32 elemCount, const CharArray& userName)
{

    if (userName.size)
    {
        if (deviceControl)
        {
            const ProfilerDeviceKit& kit = *deviceControl;
            TRACE_ROOT_STD;
            kit.gpuStreamWaiting.waitStream(kit.gpuCurrentStream, stdPass);
        }
    }

    ////

    if (currentScope)
        DEBUG_BREAK_CHECK(location != currentScope->location);

    //
    // Try to find existing node
    //

    ProfilerNode* thisNode = 0;

    if (currentScope)
    {
        ProfilerNode* p = currentScope->lastChild;

        for (; p != 0; p = p->prevBrother)
            if (p->location == location)
                break;

        thisNode = p;
    }

    //if_not (thisNode)
    //{
    //  ProfilerNode* testNode = 0;

    //  if (currentScope)
    //  {
    //    ProfilerNode* p = currentScope->lastChild;

    //    for (; p != 0; p = p->prevBrother)
    //      if (strcmp(p->location, location) == 0)
    //        break;

    //    if (p) abort();
    //  }
    //}

    //
    // If current node is not found in childs, allocate new node (if possible).
    //

    if_not (thisNode)
    {
        ARRAY_EXPOSE(nodePool);

        if (nodeCount < nodePoolSize)
        {
            thisNode = unsafePtr(&nodePoolPtr[nodeCount], 1);
            ++nodeCount;
            thisNode->init(location);

            if (currentScope)
            {
                thisNode->prevBrother = currentScope->lastChild;
                currentScope->lastChild = thisNode;
            }
        }
    }

    //
    // Update information
    //

    if (thisNode)
    {
        thisNode->totalElemCount += elemCount;
        thisNode->counter += 1;
        thisNode->userName = userName;
    }

    //
    // Save current scope, change the scope.
    //

    scope.prevCurrentScope = currentScope;
    currentScope = thisNode;

    ////

    scope.startMoment = timer.moment();

}

//================================================================
//
// ProfilerImpl::leave
//
//================================================================

inline void ProfilerImpl::leave(ProfilerScopeEx& scope)
{

    if (currentScope)
    {
        if (currentScope->userName.size)
        {
            if (deviceControl)
            {
                const ProfilerDeviceKit& kit = *deviceControl;
                TRACE_ROOT_STD;
                kit.gpuStreamWaiting.waitStream(kit.gpuCurrentStream, stdPass);
            }
        }
    }

    //
    // Measure time
    //

    ProfilerMoment endMoment = timer.moment();
    ProfilerMoment scopeTime = endMoment - scope.startMoment;

    //
    // Update current scope
    //

    if (currentScope)
        currentScope->totalTimeSum += scopeTime;

    //
    // Restore scope
    //

    currentScope = scope.prevCurrentScope;

}

//================================================================
//
// ProfilerImpl::getCurrentNodeLink
//
//================================================================

void ProfilerImpl::getCurrentNodeLink(ProfilerNodeLink& result)
{
    nodeRefOwner.connect(result.profilerChild);

    COMPILE_ASSERT(sizeof(result.reference) >= sizeof(NodePtr));
    (NodePtr&) result.reference = currentScope;
}

//================================================================
//
// ProfilerImpl::addDeviceTime
//
//================================================================

inline void ProfilerImpl::addDeviceTime(ProfilerNode* node, float32 deviceTime, float32 overheadTime)
{
    if (node)
    {
        node->deviceNodeTime += deviceTime;
        node->deviceNodeOverheadTime += overheadTime;
    }
}

//================================================================
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//----------------------------------------------------------------
//
// ProfilerThunk
//
//----------------------------------------------------------------
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//================================================================

//================================================================
//
// ProfilerThunk::enterFunc
// ProfilerThunk::leaveFunc
// ProfilerThunk::addDeviceTimeFunc
//
//================================================================

void ProfilerThunk::enterFunc(Profiler& profiler, ProfilerScope& scope, TraceLocation location)
{
    ProfilerThunk& that = static_cast<ProfilerThunk&>(profiler);

    COMPILE_ASSERT(sizeof(ProfilerScopeEx) <= sizeof(ProfilerScope));
    ProfilerScopeEx& scopeEx = (ProfilerScopeEx&) scope;

    that.impl.enter(scopeEx, location, 0, 0);
}

//----------------------------------------------------------------

void ProfilerThunk::enterExFunc(Profiler& profiler, ProfilerScope& scope, TraceLocation location, uint32 elemCount, const CharArray& userName)
{
    ProfilerThunk& that = static_cast<ProfilerThunk&>(profiler);

    COMPILE_ASSERT(sizeof(ProfilerScopeEx) <= sizeof(ProfilerScope));
    ProfilerScopeEx& scopeEx = (ProfilerScopeEx&) scope;

    that.impl.enter(scopeEx, location, elemCount, userName);
}

//----------------------------------------------------------------

void ProfilerThunk::leaveFunc(Profiler& profiler, ProfilerScope& scope)
{
    ProfilerThunk& that = static_cast<ProfilerThunk&>(profiler);

    COMPILE_ASSERT(sizeof(ProfilerScopeEx) <= sizeof(ProfilerScope));
    ProfilerScopeEx& scopeEx = (ProfilerScopeEx&) scope;

    that.impl.leave(scopeEx);
}

//----------------------------------------------------------------

void ProfilerThunk::getCurrentNodeLinkFunc(Profiler& profiler, ProfilerNodeLink& result)
{
    ProfilerThunk& that = static_cast<ProfilerThunk&>(profiler);

    that.impl.getCurrentNodeLink(result);
}

//----------------------------------------------------------------

void ProfilerThunk::addDeviceTimeFunc(const ProfilerNodeLink& node, float32 deviceTime, float32 overheadTime)
{
    COMPILE_ASSERT(sizeof(NodePtr) <= sizeof(node.reference));
    NodePtr nodePtr = (const NodePtr&) node.reference;

    if (node.profilerChild.getParent())
        ProfilerImpl::addDeviceTime(nodePtr, deviceTime, overheadTime);
}
