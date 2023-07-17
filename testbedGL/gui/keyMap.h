#include <unordered_map>

#include "lib/keys/keyBase.h"
#include "lib/signalSupport/parseKey.h"
#include "baseInterfaces/actionDefs.h"

//================================================================
//
// KeyMap
//
// May throw exceptions.
//
//================================================================

class KeyMap
{

public:

    struct KeyHash
    {
        size_t operator()(const KeyRec& r) const
        {
            return r.modifiers + size_t{KeyModifier::Max + 1} * r.code;
        }
    };

    struct KeyCompare
    {
        bool operator()(const KeyRec& a, const KeyRec& b) const
        {
            return a.code == b.code && a.modifiers == b.modifiers;
        }
    };

    void clearMemory() {map.clear();}

    void reinitTo(size_t n)
    {
        map.clear();
        map.max_load_factor(0.85f);
        map.reserve(n);
    }

    // Returns false if key parsing fails.
    bool insert(CharArray key, ActionId id)
    {
        KeyRec rec;
        ensure(parseKey(key, rec));
        map.emplace(rec, id);
        return true;
    }

    bool find(const KeyRec& key, ActionId& result)
    {
        auto p = map.find(key);
        ensure(p != map.end());
        result = p->second;
        return true;
    }

private:

    std::unordered_map<KeyRec, ActionId, KeyHash, KeyCompare> map;

};
