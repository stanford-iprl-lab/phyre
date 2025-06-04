include "shared.thrift"

namespace cpp collision

typedef shared.Error_message Err_msg,

// This is an atom of actual level database. The whole file is stored on disk.
struct Collision {
  1: required i32 bodyId1,
  2: required i32 bodyId2,
  3: required i32 timestep,
}

struct CollisionCollection {
  1: optional list<Collision> collisions,
}

const i32 COLLISION_SIZE = 3