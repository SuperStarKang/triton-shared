#ifndef MLIR_DIALECT_PIM_IR_PIM_DIALECT_H_
#define MLIR_DIALECT_PIM_IR_PIM_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Enums must be included before attribute declarations that reference them.
#include "triton-shared/Dialect/PIM/IR/PIMEnums.h.inc"

// Dialect class declaration.
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h.inc"

// Attribute class declarations.
#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.h.inc"

// Op class declarations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMOps.h.inc"

#endif // MLIR_DIALECT_PIM_IR_PIM_DIALECT_H_
