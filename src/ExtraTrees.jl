module ExtraTrees

using Distributions

export extraTrees
export predict

abstract BinaryTree

type Node <: BinaryTree
  left::BinaryTree
  right::BinaryTree
  col::Int32
  cut::Float64
end

type Leaf <: BinaryTree
  value::Float32
end

type RegressionET
  trees::Vector{BinaryTree}
  ntry::Int64
  nodesize::Int64
  nrandomcuts::Int64
end

type Sampler
  x::Vector{Int64}
  nextIndex::Int64
  Sampler(x) = new(x, 1)
end

function hasNext(s::Sampler)
  return s.nextIndex <= length(s.x)
end

function reset!(s::Sampler)
  s.nextIndex = 1
end

function next!(s::Sampler)
  i = rand( 0:(length(s.x) - s.nextIndex) )
  tmp = s.x[s.nextIndex + i]
  s.x[s.nextIndex + i] = s.x[s.nextIndex]
  s.x[s.nextIndex]     = tmp
  s.nextIndex += 1
  return tmp
end

type RegressionData
  x::Matrix{Float64}
  y::Vector{Float64}
  y2::Vector{Float64}
  N::Int
  
  RegressionData(x,y) = new(x, y, y.^2, length(y))
end

type VarScorer
  ysum::Float64
  y2sum::Float64
  n::Int64
end

function init!(scorer::VarScorer, data::RegressionData, ids)
  scorer.ysum  = 0.0
  scorer.y2sum = 0.0
  scorer.n     = length(ids)
  for id = ids
    scorer.ysum  += data.y[id]
    scorer.y2sum += data.y2[id]
  end
end

function isConstant(scorer::VarScorer)
  scorer.y2sum / scorer.n - (scorer.ysum / scorer.n) ^ 2 < 1e-7
end

## return variance for the given cut
function getScore(scorer::VarScorer, data::RegressionData, ids, col::Int64, cut::Float64)
  ## finding columns var
  ysumLeft  = 0.0
  y2sumLeft = 0.0
  countLeft = 0

  for id in ids
    if data.x[id, col] < cut
      ysumLeft  += data.y[id]
      y2sumLeft += data.y2[id]
      countLeft += 1
    end
  end
  countRight = length(ids) - countLeft
  varLeft    = y2sumLeft / countLeft - (ysumLeft / countLeft)^2
  varRight   = (scorer.y2sum - y2sumLeft) / countRight - ((scorer.ysum - ysumLeft) / countRight) ^ 2
  var = countLeft * varLeft + countRight * varRight
  return var
end

function extrema(x::Matrix{Float64}, ids, col::Int64)
  min = Inf
  max = -Inf
  for id = ids
    xv = x[id,col]
    if xv < min
      min = xv
    end
    if xv > max
      max = xv
    end
  end
  (min,max)
end

function trainTree(et::RegressionET, data::RegressionData, ids, sampler::Sampler, scorer::VarScorer)
  if length(ids) < et.nodesize
    return Leaf(mean(data.y[ids]))
  end

  ## initialize scorer for current node
  init!(scorer, data, ids)
  
  if isConstant(scorer)
    return Leaf(scorer.ysum / length(ids))
  end
  
  bestVar = Inf
  bestCol = -1
  bestCut = 0.0
  
  ## random columns
  reset!(sampler)
  
  ## check K columns
  K = 0
  while hasNext(sampler)
    col = next!(sampler)
    range = extrema(data.x, ids, col)
    if range[1] == range[2]
      continue
    end
    for ncut = 1:et.nrandomcuts
      ## random cut
      cut = range[1] + (range[2] - range[1]) * rand()

      var = getScore(scorer, data, ids, col, cut)
      if var < bestVar
        bestVar = var
        bestCol = col
        bestCut = cut
      end
    end
    K += 1
    if K >= et.ntry
      break
    end
  end
  ## splitting samples
  idsLeft  = Int64[]
  idsRight = Int64[]
  for i = ids
    if data.x[i, bestCol] < bestCut
      push!(idsLeft, i)
    else
      push!(idsRight, i)
    end
  end
  
  leftNode  = trainTree(et, data, idsLeft, sampler, scorer)
  rightNode = trainTree(et, data, idsRight, sampler, scorer)
  
  return Node(leftNode, rightNode, bestCol, bestCut)
end

function extraTrees(x::Matrix{Float64}, y::Vector{Float64};
                            ntree     = 500, 
                            ntry      = int( max(1, floor(size(x,1) / 3)) ),
                            nrandomcuts = 1,
                            nodesize  = 5)
  data  = RegressionData(x, y)
  trees = BinaryTree[]
  et    = RegressionET(trees, ntry, nodesize, nrandomcuts)
  scorer  = VarScorer(0.0, 0.0, 0)
  sampler = Sampler([1:size(x,2)])
  for i in 1:ntree
    push!(trees, trainTree(et, data, 1:length(y), sampler, scorer) )
  end
  return et
end

function predict(tree::Leaf, x::Vector{Float64})
  return tree.value
end

function predict(tree::Node, x::Vector{Float64})
  if x[tree.col] < tree.cut
    return predict(tree.left, x)
  else
    return predict(tree.right, x)
  end
end

function predict(et::RegressionET, x::Vector{Float64})
  sum = 0.0
  for tree = et.trees
    sum += predict(tree, x)
  end
  return sum / length(et.trees)
end

function predict(et::RegressionET, Xnew::Matrix{Float64})
  map(
    i -> predict(et, vec(Xnew[i,:])),
    1:size(Xnew, 1)
  )
end

end # module
