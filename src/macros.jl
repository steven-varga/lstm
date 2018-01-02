#=
_______________________________________________________________________________

   VARGA CONSULTING
  __________________

   [2010] - [2018] Varga Consulting
   All Rights Reserved.
  ______________________________________________________________________________
  NOTICE: All information contained herein is, and remains the property of Varga
  Consulting and its suppliers, if  any. The intellectual and technical concepts
  contained herein are proprietary to Varga Consulting and its suppliers and may
  be covered by U.S. and Foreign  Patents, patents in process, and are protected
  by  trade  secret or  copyright  law.  Dissemination  of this  information  or
  reproduction  of this  material  is strictly  forbidden  unless prior  written
  permission is obtained from Varga Consulting.
  ______________________________________________________________________________

  Copyright (c) <2018> < Copyright © 2018 Steven Varga, Toronto, On>

  Contact: Steven Varga
           <steven@vargaconsulting.ca>
           2018 Toronto, On Canada
  ______________________________________________________________________________
  =#

using Distributions

svec{T}(M::Matrix{T} ) =  unsafe_wrap(Array{T}, pointer(M,1), length(M),false)
svec{T}(v::Vector{T}, i::Int64,j::Int64 ) =  unsafe_wrap(Array{T}, pointer(v,i), j,false)
smat{T}(v::Vector{T}, i::Int64,j::Int64, k::Int64 ) = unsafe_wrap(Array{T}, pointer(v,i), (j,k), false)

export svec,smat, @define,@attach,@similar,@rand!,@fill!,@shared,@set

macro set( st, fields... )
	block = Expr(:block)
	
	for f in fields
		push!(block.args, :($st.$f = $f) )
    end
	
	esc(:($block))
end
macro attach( st, fields... )
	block = Expr(:block)
	
	for f in fields
		push!(block.args, :($f = $st.$f) )
    end
	
	esc(:($block))
end
macro define(T, fields... )
	block = Expr(:block)
	for f in fields
		push!(block.args, :($f ::$T ))
    end
	return esc(:($block))
end
macro similar(value, vars... )
	block = Expr(:block)
	for v in vars
		push!(block.args, :($v = similar($value) ))
    end
	return esc(:($block))
end
macro rand!(dist, vars... )
	block = Expr(:block)
	for var in vars
		push!(block.args, :(rand!($dist,$var) ))
    end
	return esc(:($block))
end
macro fill!(value, vars... )
	block = Expr(:block)
	for var in vars
		push!(block.args, :(fill!($var,$value) ))
    end
	return esc(:($block))
end


macro shared(T,ref, vars... )
	block = Expr(:block )
	i=:0
	## calculate size of storage vector
	for v in vars
		vtype = v.args[1]
		m,n,k = v.args[2],v.args[3],length(v.args)
	    i =	if vtype == :Matrix
				:($i + ($k-3)*$m*$n)
			elseif vtype == :Vector
				:($i + ($k-2)*$m)
			end
    end
	# define with given type
	push!( block.args, :( $ref = Vector{$T}($i) ) )
	
	#
	i = :1
	for v in vars
		vtype = v.args[1]
		if vtype == :Matrix
            m,n = v.args[2],v.args[3]
			for mat in v.args[4:end]
				push!(block.args,:($mat = smat($ref,$i,$m,$n)))
				i = :( $i + $m*$n)
			end
		elseif vtype == :Vector
            m = v.args[2]
			for mat in v.args[3:end]
				push!(block.args,:($mat = svec($ref,$i,$m)))
				i = :( $i + $m)
			end
		end
	
    end


	return esc(:($block))
end

#= 
N,M,K=1,1,1

@shared(Float32,θ, 
	  Matrix(N,M, a,b,c,d,f),  Matrix(N,N, j,k,l), Vector(N, m,n,o) )
fill!(θ,1)
cumsum!(θ,θ)
println( θ )
=#
