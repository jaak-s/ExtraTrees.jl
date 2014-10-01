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
  ntry::Int
  nodesize::Int
  trees::Vector{BinaryTree}
end

type Sampler
  x::Vector{Int}
  nextIndex::Int
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

function trainTree(et::RegressionET, data::RegressionData, ids, sampler::Sampler)
  if length(ids) < et.nodesize
    return Leaf(mean(data.y[ids]))
  end

  y  = data.y[ids]
  y2 = data.y2[ids]
  
  ysum  = sum(y)
  y2sum = sum(y2)
  varTotal = y2sum / length(ids) - (ysum / length(ids)) ^ 2
  if varTotal < 1e-7
    return Leaf(ysum / length(ids))
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
    range = extrema(data.x[ids,col])
    if range[1] == range[2]
      continue
    end
    ## random cut
    cut = range[1] + (range[2] - range[1]) * rand()
    ## finding columns var
    ysumLeft  = 0.0
    y2sumLeft = 0.0
    countLeft = 0
    for i in 1:length(ids)
      if data.x[ids[i], col] < cut
        ysumLeft  += y[i]
        y2sumLeft += y2[i]
        countLeft += 1
      end
    end
    countRight = length(ids) - countLeft
    varLeft    = y2sumLeft / countLeft - (ysumLeft / countLeft)^2
    varRight   = (y2sum - y2sumLeft) / countRight - ((ysum - ysumLeft) / countRight) ^ 2
    var = countLeft * varLeft + countRight * varRight
    if var < bestVar
      bestVar = var
      bestCol = col
      bestCut = cut
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
  
  leftNode  = trainTree(et, data, idsLeft, sampler)
  rightNode = trainTree(et, data, idsRight, sampler)
  
  return Node(leftNode, rightNode, bestCol, bestCut)
end

function extraTrees(x::Matrix{Float64}, y::Vector{Float64};
                            ntree    = 500, 
                            ntry     = max(1, floor(size(x,1) / 3)),
                            nodesize = 5)
  data  = RegressionData(x, y)
  trees = BinaryTree[]
  et    = RegressionET(ntry, nodesize, trees)
  sampler = Sampler([1:size(x,2)])
  for i in 1:ntree
    push!(trees, trainTree(et, data, 1:length(y), sampler) )
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
