#include "cfgTree.h"

#include "hashTools.h"
#include "storage/rememberCleanup.h"
#include "cfgVars/types/stringStorage.h"
#include "userOutput/printMsg.h"
#include "errorLog/debugBreak.h"
#include "compileTools/blockExceptionsSilent.h"

namespace cfgTree {

//================================================================
//
// getHash
//
//================================================================

Hash getHash(const StringRef& str)
{
    return Hash(hashBytes(str.ptr, str.size * sizeof(*str.ptr)));
}

//================================================================
//
// NameStorage
//
//================================================================

class NameStorage
{

public:

    sysinline operator NameRef() const
    {
        return {hash, str};
    }

    sysinline StringRef getStr() const
    {
        return str;
    }

    sysinline bool operator ==(const NameRef& that) const
    {
        return (hash == that.hash) && (str == that.str);
    }

    sysinline void operator =(const NameRef& that) may_throw
    {
        str = that.str; // May throw
        hash = that.hash;
    }

    sysinline void clear()
    {
        str.clear();
        hash = getHash({});
    }

    sysinline void dealloc()
    {
        str.dealloc();
        hash = getHash({});
    }

    sysinline void swap(NameStorage& that)
    {
        str.swap(that.str);
        exchangeByCopying(hash, that.hash);
    }

    sysinline size_t dynamicBytes() const
    {
        return str.capacity() * sizeof(Char);
    }

private:

    StringStorage str;
    Hash hash = getHash({});

};

//================================================================
//
// DataStorage
//
//================================================================

class DataStorage
{

public:

    sysinline bool hasData() const
    {
        return isSet;
    }

    sysinline GetDataResult get() const
    {
        return {value, comment, blockComment};
    }

    template <bool trackChange, typename SpeedCheck>
    sysinline void set(const SetDataArgs& args, const SpeedCheck& speedCheck) may_throw
    {
        isSet = true;

        if (trackChange)
        {
            bool equal = true;

            check_flag(value == args.value, equal);
            check_flag(comment == args.comment, equal);
            check_flag(blockComment == args.blockComment, equal);

            if (equal)
                return;

            changed = true;
        }

        speedCheck(args.value.size <= value.capacity());
        value = args.value;

        speedCheck(args.comment.size <= comment.capacity());
        comment = args.comment;

        speedCheck(args.blockComment.size <= blockComment.capacity());
        blockComment = args.blockComment;
    }

    sysinline void clear()
    {
        isSet = false;
        changed = false;
        value.clear();
        comment.clear();
        blockComment.clear();
    }

    sysinline void dealloc()
    {
        isSet = false;
        changed = false;
        value.dealloc();
        comment.dealloc();
        blockComment.dealloc();
    }

    sysinline void swap(DataStorage& that)
    {
        exchange(isSet, that.isSet);
        exchange(changed, that.changed);
        value.swap(that.value);
        comment.swap(that.comment);
        blockComment.swap(that.blockComment);
    }

    sysinline size_t dynamicBytes() const
    {
        return sizeof(Char) * (value.capacity() + comment.capacity() + blockComment.capacity());
    }

    sysinline bool isChanged() const
    {
        return changed;
    }

    sysinline void clearChanged()
    {
        changed = false;
    }

private:

    bool isSet = false;
    bool changed = false;

    StringStorage value;
    StringStorage comment;
    StringStorage blockComment;

};

//================================================================
//
// NodeImpl
//
//================================================================

class NodeImpl : public Node
{

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    static constexpr bool allocReport = false;
    static constexpr bool debugChecks = false;

    NameStorage name;
    DataStorage data;

    //
    // The children are stored as a circular linked list where each element
    // points to the next one. The last element of the list points to the first one.
    // For example, a list with only one element is represented as an element
    // that points to itself.
    //
    // The base pointers refer to the last element of the list.
    // If the list is empty, the pointer is NULL.
    //
    // A node could be only in one of these two lists at a time.
    //
    // When moved to the shadow list, the contents of a node is cleared, except for its name:
    // all of its children are cleared (moved to its own shadow list).
    //

    NodeImpl* lastChild = nullptr;
    NodeImpl* lastShadow = nullptr;

    NodeImpl* next = nullptr;

public:

    //----------------------------------------------------------------
    //
    // Init & deinit.
    //
    //----------------------------------------------------------------

    sysinline NodeImpl()
    {
    }

    sysinline ~NodeImpl()
    {
        dealloc();
    }

    void clearAllButName()
    {
        data.clear();
        clearChildren();
        unlink();
    }

    virtual void clearAll()
    {
        name.clear();
        clearAllButName();
    }

    void dealloc()
    {
        name.dealloc();
        data.dealloc();
        deallocChildren();
        unlink();
    }

    //----------------------------------------------------------------
    //
    // forEachNode
    //
    //----------------------------------------------------------------

    template <typename Action>
    static sysinline void forEachNode(NodeImpl* lastNode, const Action& action)
    {
        if_not (lastNode)
            return;

        ////

        auto first = lastNode->next;
        auto current = first;
        auto prev = lastNode;

        do
        {
            // Read before the action so it could destroy the current node.
            auto next = current->next;

            action(current, prev);

            prev = current;
            current = next;
        }
        while (current != first);
    }

    //----------------------------------------------------------------
    //
    // Children API.
    //
    //----------------------------------------------------------------

    virtual bool hasChildren() const
    {
        return lastChild != nullptr;
    }

    ////

    virtual void forAllChildren(NodeHandler& handler)
    {
        forEachNode
        (
            lastChild,
            [&] (auto* ptr, auto) {handler(*ptr);}
        );
    }

    ////

    virtual void deallocChildren()
    {
        auto action = [] (auto* ptr, auto) {delete ptr;};

        forEachNode(lastChild, action);
        lastChild = nullptr;

        forEachNode(lastShadow, action);
        lastShadow = nullptr;
    }

    ////

    virtual void clearChildren()
    {
        auto action = [&] (auto* ptr, auto)
        {
            ptr->clearAllButName();
            addChild(lastShadow, ptr);
        };

        forEachNode(lastChild, action);
        lastChild = nullptr;
    }

    //----------------------------------------------------------------
    //
    // addChild
    //
    //----------------------------------------------------------------

    static sysinline void addChild(NodeImpl*& lastNode, NodeImpl* newNode)
    {
        if_not (lastNode)
        {
            newNode->next = newNode;
        }
        else
        {
            auto first = lastNode->next;
            lastNode->next = newNode;
            newNode->next = first;
        }

        lastNode = newNode;
    }

    //----------------------------------------------------------------
    //
    // findChild
    //
    //----------------------------------------------------------------

    static sysinline NodeImpl* findChildEx(NodeImpl* lastNode, const NameRef& name)
    {
        NodeImpl* result = nullptr;

        auto action = [&] (auto* ptr, auto)
        {
            if (ptr->name == name)
                result = ptr;
        };

        forEachNode(lastNode, action);
        return result;
    }

    ////

    virtual Node* findChild(const NameRef& name)
    {
        return findChildEx(lastChild, name);
    }

    //----------------------------------------------------------------
    //
    // findAndRemoveChild
    //
    //----------------------------------------------------------------

    static sysinline NodeImpl* findAndRemoveChild(NodeImpl*& lastNode, const NameRef& name)
    {
        NodeImpl* foundCurr = nullptr;
        NodeImpl* foundPrev = nullptr;

        auto action = [&] (auto* curr, auto* prev)
        {
            if (curr->name == name)
            {
                foundCurr = curr;
                foundPrev = prev;
            }
        };

        forEachNode(lastNode, action);

        if_not (foundCurr)
            return nullptr;

        ////

        auto prev = foundPrev;
        auto curr = foundCurr;
        auto next = curr->next;

        prev->next = next; // remove
        curr->next = nullptr; // just for debug

        ////

        if (lastNode == curr)
        {
            lastNode = (curr == next) ? // single-element list?
                nullptr : prev;
        }

        if (debugChecks)
            DEBUG_BREAK_IF(findChildEx(lastNode, name));

        return curr;
    }

    //----------------------------------------------------------------
    //
    // findOrCreateChild
    //
    //----------------------------------------------------------------

    virtual Node* findOrCreateChild(const NameRef& name) may_throw
    {
        auto* findResult = findChildEx(lastChild, name);

        if (findResult)
            return findResult;

        ////

        auto* shadow = findAndRemoveChild(lastShadow, name);

        if (shadow)
        {
            addChild(lastChild, shadow);
            return shadow;
        }

        ////

        reportAlloc(STR("Node alloc"));

        auto newChild = new NodeImpl; // may throw
        REMEMBER_CLEANUP_EX(newChildClean, delete newChild);

        newChild->name = name;

        ////

        newChildClean.cancel();

        addChild(lastChild, newChild);

        return newChild;
    }

    //----------------------------------------------------------------
    //
    // removeChild
    //
    //----------------------------------------------------------------

    virtual bool removeChild(const NameRef& name)
    {
        auto* node = findAndRemoveChild(lastChild, name);
        ensure(node);

        if (debugChecks)
            DEBUG_BREAK_IF(findChildEx(lastShadow, name));

        node->clearAllButName();
        addChild(lastShadow, node);

        return true;
    }

    //----------------------------------------------------------------
    //
    // Name
    //
    //----------------------------------------------------------------

    virtual StringRef getName() const
        {return name.getStr();}

    virtual void clearName()
        {name.clear();}

    //----------------------------------------------------------------
    //
    // Data.
    //
    //----------------------------------------------------------------

    virtual bool hasData() const
        {return data.hasData();}

    virtual void clearData()
        {return data.clear();}

    virtual GetDataResult getData() const
        {return data.get();}

    ////

    template <bool trackChange>
    sysinline void setDataGeneric(const SetDataArgs& args) may_throw
    {
        if_not (allocReport)
        {
            data.set<trackChange>(args, [&] (auto) {});
        }
        else
        {
            bool fast = true;

            data.set<trackChange>(args, [&] (bool ok) {if (!ok) fast = false;});

            if_not (fast)
                reportAlloc(STR("Data alloc"));
        }
    }

    ////

    virtual void setData(const SetDataArgs& args) may_throw
        {setDataGeneric<false>(args);}

    virtual void setDataEx(const SetDataArgs& args) may_throw
        {setDataGeneric<true>(args);}

    //----------------------------------------------------------------
    //
    // Delta buffer API.
    //
    //----------------------------------------------------------------

    virtual void moveFrom(Node& other)
    {
        clearAll();
        absorb(other);
    }

    ////

    virtual bool absorb(Node& other)
    {
        NodeImpl& that = (NodeImpl&) other;

        this->name.clear();
        that.name.clear();

        this->data.clear();
        that.data.clear();

        absorbChildren(*this, that);

        return true;
    }

    //----------------------------------------------------------------
    //
    // absorbChildren
    //
    // Strangely it doesn't allocate anything and never throws.
    //
    //----------------------------------------------------------------

    static sysinline void absorbChildren(NodeImpl& L, NodeImpl& R)
    {
        auto action = [&] (auto* Rchild, auto)
        {
            if (debugChecks)
                DEBUG_BREAK_IF(findChildEx(R.lastShadow, Rchild->name));

            ////

            auto* Lchild = findChildEx(L.lastChild, Rchild->name);

            if_not (Lchild)
            {
                Lchild = findAndRemoveChild(L.lastShadow, Rchild->name); // Find in the pool.

                if (Lchild)
                    addChild(L.lastChild, Lchild);
            }

            if_not (Lchild)
            {
                addChild(L.lastChild, Rchild);
                return;
            }

            //
            // The name is equal, absorb data and children.
            //

            Lchild->data.swap(Rchild->data);

            absorbChildren(*Lchild, *Rchild);

            //
            // Clear the node and save it the pool.
            //

            Rchild->clearAllButName();
            addChild(R.lastShadow, Rchild);
        };

        forEachNode(R.lastChild, action);
        R.lastChild = nullptr;
    }

    //----------------------------------------------------------------
    //
    // Debugging tools.
    //
    //----------------------------------------------------------------

    sysinline void unlink()
    {
        next = nullptr;
    }

    sysnoinline void reportAlloc(StringRef action)
    {
        if (allocReport)
            action = action;
    }

    //----------------------------------------------------------------
    //
    // allocatedBytes
    //
    //----------------------------------------------------------------

    virtual size_t allocatedBytes() const
    {
        size_t amount = 0;

        amount += sizeof(*this);
        amount += name.dynamicBytes();
        amount += data.dynamicBytes();

        auto action = [&] (auto* ptr, auto) {amount += ptr->allocatedBytes();};

        forEachNode(lastChild, action);
        forEachNode(lastShadow, action);

        return amount;
    }

    //----------------------------------------------------------------
    //
    // clearAllDataChangedFlags
    //
    //----------------------------------------------------------------

    virtual void clearAllDataChangedFlags()
    {
        data.clearChanged();

        auto action = [&] (auto* ptr, auto) {ptr->clearAllDataChangedFlags();};
        forEachNode(lastChild, action);
    }

    //----------------------------------------------------------------
    //
    // generateUpdate
    //
    //----------------------------------------------------------------

    struct PathFrame
    {
        NodeImpl* node;
        PathFrame* next;
    };

    ////

    void generateUpdateEx(PathFrame& firstFrame, PathFrame& lastFrame, Node& dstTree)
    {

        //
        // If the data is changed, create the entire path in dstTree and set the value.
        // Process range (firstFrame, lastFrame].
        //

        if (data.isChanged())
        {
            auto* frame = &firstFrame;
            auto* dst = &dstTree;

            while (frame != &lastFrame)
            {
                frame = frame->next;
                dst = dst->findOrCreateChild(frame->node->name);
            }

            dst->setData(frame->node->getData());
        }

        //
        // Traverse subtrees, keeping the path.
        //

        auto action = [&] (auto* child, auto)
        {
            PathFrame newFrame{child, nullptr};
            lastFrame.next = &newFrame;

            child->generateUpdateEx(firstFrame, newFrame, dstTree);
        };

        forEachNode(lastChild, action);
    }

    ////

    virtual void generateUpdate(Node& dstTree) may_throw
    {
        PathFrame firstFrame{nullptr, nullptr};

        generateUpdateEx(firstFrame, firstFrame, dstTree);
    }

};

////

UniquePtr<Node> Node::create() {return makeUnique<NodeImpl>();}

//----------------------------------------------------------------

}
